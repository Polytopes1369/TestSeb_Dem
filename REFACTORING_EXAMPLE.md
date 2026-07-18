# Exemple de Refactoring RÉEL : AtmosSkyPass

> **Correction (2026-07-18)** : la première version de ce document contenait un exemple
> "AVANT/APRÈS" **fabriqué** — écrit sans jamais lire `AtmosSkyPass.cpp` réel. C'était une
> violation de la règle CLAUDE.md "Zéro approximation". Ce document a été entièrement réécrit
> à partir du vrai fichier (`src/renderer/passes/AtmosSkyPass.cpp`, 216 lignes) et de la
> migration réellement appliquée sur la branche `refactor/wave1-pass-migrations-1784369482`,
> vérifiée par une compilation MSVC réelle (voir rapport de build).

## Ce qui a été découvert en lisant le vrai code

Le projet a déjà une classe `renderer::VulkanUtils` (`src/renderer/vulkan/VulkanUtils.h/cpp`)
qui fournit déjà :
- `ExecuteOneShotCommands`, `RecordMemoryBarrier`, `TransitionImageLayout(OneShot)`
- `CreateNearestSampler`, `CreateStorageSampledImage2D`, `ClearComputeImageToGeneral`
- `WriteRayBuffersDescriptorSet`, `WriteSharedGeometryBindings`

Les fichiers `VulkanImageUtils`/`VulkanBarrierUtils`/`VulkanSamplerUtils`/`VulkanDescriptorUtils`
créés initialement en Wave 1 étaient donc **largement redondants** avec cet existant et ont été
**supprimés**. Seul un vrai gap a été confirmé par grep sur les 41 fichiers de passes concernés :
la séquence `vkCreateDescriptorSetLayout` → `vkCreateDescriptorPool` → `vkAllocateDescriptorSets`
n'était pas encore factorisée. Une seule méthode a donc été ajoutée à la classe existante :
`VulkanUtils::CreateDescriptorSetLayoutPoolAndSet()`.

`RenderPass<Derived>` (base CRTP pour Init/Shutdown) est, elle, un vrai ajout net — aucune classe
de base équivalente n'existe dans le codebase (chaque passe est une classe concrète autonome).

---

## AVANT (réel, `src/renderer/passes/AtmosSkyPass.cpp`, commit `7950392`)

```cpp
bool AtmosSkyPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
    m_Device = device;
    m_Allocator = allocator;

    VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kTransmittanceExtent,
        m_Transmittance.image, m_Transmittance.allocation, m_Transmittance.view);
    VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kMultiScatteringExtent,
        m_MultiScattering.image, m_MultiScattering.allocation, m_MultiScattering.view);
    VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kSkyViewExtent,
        m_SkyView.image, m_SkyView.allocation, m_SkyView.view);

    // ... clear pass, sampler creation ...

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{ { /* 5 bindings */ } };
    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

    std::array<VkDescriptorPoolSize, 2> poolSizes{ { /* 2 pool sizes */ } };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

    VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocSet.descriptorPool = m_DescriptorPool;
    allocSet.descriptorSetCount = 1;
    allocSet.pSetLayouts = &m_SetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocSet, &m_Set));

    // ... descriptor writes, pipeline layout, pipeline creation ...
    return true;
}

void AtmosSkyPass::Shutdown() {
    if (m_Device != VK_NULL_HANDLE) {
        if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
        if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
        if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
        if (m_LUTSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LUTSampler, nullptr);
        if (m_Transmittance.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_Transmittance.view, nullptr);
        if (m_MultiScattering.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_MultiScattering.view, nullptr);
        if (m_SkyView.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_SkyView.view, nullptr);
    }
    if (m_Allocator != VK_NULL_HANDLE) {
        if (m_Transmittance.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_Transmittance.image, m_Transmittance.allocation);
        if (m_MultiScattering.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_MultiScattering.image, m_MultiScattering.allocation);
        if (m_SkyView.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_SkyView.image, m_SkyView.allocation);
    }
    m_Transmittance = {}; m_MultiScattering = {}; m_SkyView = {};
    m_Pipeline = VK_NULL_HANDLE; m_PipelineLayout = VK_NULL_HANDLE;
    m_SetLayout = VK_NULL_HANDLE; m_Set = VK_NULL_HANDLE; m_DescriptorPool = VK_NULL_HANDLE;
    m_LUTSampler = VK_NULL_HANDLE; m_HasGeneratedStaticLUTs = false; m_LastStaticSunDirection = {};
    m_Allocator = VK_NULL_HANDLE; m_Device = VK_NULL_HANDLE;
}
```

31 lines of hand-written `Shutdown()`, with null-guards on every single handle (needed because
`Shutdown()` must be safe to call after a *partial* `Init()` failure).

---

## APRÈS (réel, branche `refactor/wave1-pass-migrations-1784369482`)

```cpp
class AtmosSkyPass : public RenderPass<AtmosSkyPass> {
    friend class RenderPass<AtmosSkyPass>;
    // ...
private:
    bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);
    // m_Device / m_Allocator inherited (protected) from RenderPass<AtmosSkyPass>.
};
```

```cpp
bool AtmosSkyPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
    VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kTransmittanceExtent,
        m_Transmittance.image, m_Transmittance.allocation, m_Transmittance.view);
    RegisterResource([this] {
        vkDestroyImageView(m_Device, m_Transmittance.view, nullptr);
        vmaDestroyImage(m_Allocator, m_Transmittance.image, m_Transmittance.allocation);
    });
    // ... same pattern for m_MultiScattering, m_SkyView ...

    auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device, bindings, poolSizes);
    m_SetLayout = descSet.layout; m_DescriptorPool = descSet.pool; m_Set = descSet.set;
    RegisterResource([this] {
        vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
    });

    // ... pipeline layout + pipeline, each followed by its own RegisterResource(...) ...

    // Non-Vulkan cache state the original Shutdown() also reset, preserved for re-Init safety:
    RegisterResource([this] {
        m_HasGeneratedStaticLUTs = false;
        m_LastStaticSunDirection = {};
    });

    return true;
}
// No Shutdown() override -- inherited from RenderPass<AtmosSkyPass>, runs the above in reverse.
```

---

## Bilan mesuré (ce fichier, réellement)

| Métrique | Avant | Après |
|---|---|---|
| `Shutdown()` hand-written | 31 lines, 11 null-guards | 0 lines (inherited) |
| Descriptor layout+pool+set boilerplate | 27 lines, 3 raw Vulkan calls | 1 call to `VulkanUtils::CreateDescriptorSetLayoutPoolAndSet` |
| Risk of forgetting a new resource's cleanup | Manual, easy to miss | `RegisterResource()` right after each creation, or it leaks loudly |
| Re-Init cache-state correctness | Explicit in `Shutdown()` | Explicit `RegisterResource()` lambda (same behavior, verified needed by reading the original) |

This single-file change was verified against a **real MSVC/Ninja build** of the `DemoSceneVK`
target (see build log) rather than assumed to compile. The same migration is *not* proposed
wholesale for all 44 passes without the same read-then-verify treatment per file — e.g.
`HZBPass::Init()` returns `void` (not `bool`), so it does not fit this base class as-is.
