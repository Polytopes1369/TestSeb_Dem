#include "renderer/passes/VirtualTextureRenderPass.h"

#include <algorithm>
#include <format>

#include "core/Logger.h"

namespace renderer {

    bool VirtualTextureRenderPass::Init(VkDevice device, VirtualTextureManager* vtManager,
        const VirtualTextureVolumeBounds& bounds) {
        Shutdown();

        m_Device = device;
        m_VTManager = vtManager;
        m_Bounds = bounds;

        // =====================================================================================
        // Fixed top-down (Y-up) orthographic view-projection covering the WHOLE virtual texture
        // volume -- computed ONCE, reused for every page at every mip (see class comment on the
        // virtual-viewport technique for why the projection itself never needs to change).
        // =====================================================================================
        const float centerX = (bounds.worldMinXZ.x + bounds.worldMaxXZ.x) * 0.5f;
        const float centerZ = (bounds.worldMinXZ.y + bounds.worldMaxXZ.y) * 0.5f;
        const float halfWidth = std::max((bounds.worldMaxXZ.x - bounds.worldMinXZ.x) * 0.5f, 1.0e-3f);
        const float halfDepth = std::max((bounds.worldMaxXZ.y - bounds.worldMinXZ.y) * 0.5f, 1.0e-3f);

        // Margin above/below the volume's declared height range, same convention as
        // VirtualShadowMapPass::kSunNearMarginFactor -- keeps geometry sitting exactly on
        // heightMin/heightMax from being clipped by a zero-thickness near/far plane.
        const float heightRange = std::max(bounds.heightMax - bounds.heightMin, 1.0e-3f);
        const float margin = std::max(heightRange * 0.05f, 0.01f);

        const maths::vec3 eye{ centerX, bounds.heightMax + margin, centerZ };
        const maths::vec3 lookTarget{ centerX, bounds.heightMin - margin, centerZ };
        // Forward is exactly (0,-1,0) (straight down) -- an up vector along Y would be degenerate
        // (parallel to forward), so -Z is used instead, matching VirtualShadowMapPass's own
        // "pick a non-parallel up vector" rationale for its sun clipmap.
        const maths::vec3 up{ 0.0f, 0.0f, -1.0f };
        const maths::mat4 view = maths::mat4::LookAt(eye, lookTarget, up);

        const float nearPlane = margin;                     // Distance from eye down to heightMax.
        const float farPlane = margin + heightRange + margin; // Distance from eye down to heightMin - margin.
        const maths::mat4 proj = maths::mat4::OrthoVulkan(halfWidth, halfDepth, nearPlane, farPlane);

        m_VolumeViewProj = proj * view;

        LOG_INFO(std::format(
            "[VirtualTextureRenderPass] Initialized: world volume=({:.2f},{:.2f})..({:.2f},{:.2f}), "
            "height=[{:.2f},{:.2f}], {} physical pool(s), tile={}px (+{}px border).",
            bounds.worldMinXZ.x, bounds.worldMinXZ.y, bounds.worldMaxXZ.x, bounds.worldMaxXZ.y,
            bounds.heightMin, bounds.heightMax, m_VTManager->GetPhysicalPoolCount(),
            m_VTManager->GetTileSize(), m_VTManager->GetBorderSize()));

        return true;
    }

    void VirtualTextureRenderPass::Shutdown() {
        m_Device = VK_NULL_HANDLE;
        m_VTManager = nullptr;
        m_Bounds = VirtualTextureVolumeBounds{};
        m_VolumeViewProj = maths::mat4{};
        m_PendingThisFrameKeys.clear();
    }

    void VirtualTextureRenderPass::RequestPage(uint32_t x, uint32_t y, uint32_t mip) {
        m_PendingThisFrameKeys.push_back(VirtualTextureManager::PackPageKey(x, y, mip));
    }

    void VirtualTextureRenderPass::ComputePageWorldRect(uint32_t x, uint32_t y, uint32_t mip,
        maths::vec2& outMin, maths::vec2& outMax) const {
        const uint32_t gridW = std::max(1u, m_VTManager->GetGridWidth() >> mip);
        const uint32_t gridH = std::max(1u, m_VTManager->GetGridHeight() >> mip);

        const float u0 = float(x) / float(gridW);
        const float u1 = float(x + 1) / float(gridW);
        const float v0 = float(y) / float(gridH);
        const float v1 = float(y + 1) / float(gridH);

        const float worldSpanX = m_Bounds.worldMaxXZ.x - m_Bounds.worldMinXZ.x;
        const float worldSpanZ = m_Bounds.worldMaxXZ.y - m_Bounds.worldMinXZ.y;

        const float contentMinX = m_Bounds.worldMinXZ.x + worldSpanX * u0;
        const float contentMaxX = m_Bounds.worldMinXZ.x + worldSpanX * u1;
        const float contentMinZ = m_Bounds.worldMinXZ.y + worldSpanZ * v0;
        const float contentMaxZ = m_Bounds.worldMinXZ.y + worldSpanZ * v1;

        // Expand by the border's own world-space footprint: one texel of this page's CONTENT area
        // (tileSize texels wide) spans (contentMax-contentMin)/tileSize world units, so borderSize
        // texels of padding is exactly that many world units on each side. This is what makes a
        // rendered tile's border texels sample the correct NEIGHBORING page's content (see
        // ComputePageViewport's own comment on why the virtual-viewport raster is scaled in
        // content-only tileSize units, which is what makes this consistent).
        const float tileSize = float(m_VTManager->GetTileSize());
        const float borderSize = float(m_VTManager->GetBorderSize());
        const float borderWorldX = ((contentMaxX - contentMinX) / tileSize) * borderSize;
        const float borderWorldZ = ((contentMaxZ - contentMinZ) / tileSize) * borderSize;

        outMin = maths::vec2{ contentMinX - borderWorldX, contentMinZ - borderWorldZ };
        outMax = maths::vec2{ contentMaxX + borderWorldX, contentMaxZ + borderWorldZ };
    }

    void VirtualTextureRenderPass::ComputePageViewport(uint32_t x, uint32_t y, uint32_t mip,
        VkViewport& outViewport) const {
        const uint32_t gridW = std::max(1u, m_VTManager->GetGridWidth() >> mip);
        const uint32_t gridH = std::max(1u, m_VTManager->GetGridHeight() >> mip);
        const uint32_t tileSize = m_VTManager->GetTileSize();
        const uint32_t borderSize = m_VTManager->GetBorderSize();

        // Virtual-viewport technique (see class comment): the FULL fixed projection is rasterized
        // across a (gridW*tileSize) x (gridH*tileSize) raster -- CONTENT-only units, not
        // tileSizeWithBorder -- because one content-sized raster pixel is, by construction, exactly
        // one world-space texel of border padding too (ComputePageWorldRect relies on this same
        // ratio). The viewport is offset so raster column/row (x*tileSize - borderSize) lands at
        // attachment-local column/row 0, i.e. the physical tile's own border-inclusive origin.
        outViewport.x = -(float(x) * float(tileSize)) + float(borderSize);
        outViewport.y = -(float(y) * float(tileSize)) + float(borderSize);
        outViewport.width = float(gridW) * float(tileSize);
        outViewport.height = float(gridH) * float(tileSize);
        outViewport.minDepth = 0.0f;
        outViewport.maxDepth = 1.0f;
    }

    void VirtualTextureRenderPass::RecordPageUpdates(VkCommandBuffer cmd, const VirtualTexturePageRenderFn& renderFn) {
        // Fold this frame's RequestPage() reports into the dedup queue as a single batch (see
        // m_PendingThisFrameKeys' own comment).
        if (!m_PendingThisFrameKeys.empty()) {
            // Priority = mip level (see VirtualTextureStreamingCoordinator's identical rationale) --
            // a coarser page renders before a finer one.
            std::vector<float> pendingPriorities;
            pendingPriorities.reserve(m_PendingThisFrameKeys.size());
            for (uint32_t pageKey : m_PendingThisFrameKeys) {
                uint32_t x = 0, y = 0, mip = 0;
                VirtualTextureManager::UnpackPageKey(pageKey, x, y, mip);
                pendingPriorities.push_back(float(mip));
            }
            m_RequestQueue.SubmitFrameRequests(m_PendingThisFrameKeys, pendingPriorities);
            m_PendingThisFrameKeys.clear();
        }

        bool renderedAnyPage = false;
        for (uint32_t processed = 0; processed < kMaxPagesRenderedPerFrame; ++processed) {
            uint32_t key = 0;
            if (!m_RequestQueue.PopNextRequest(key)) {
                break;
            }

            uint32_t x = 0, y = 0, mip = 0;
            VirtualTextureManager::UnpackPageKey(key, x, y, mip);

            // Already rendered and still resident at exactly this mip -- the caller is simply
            // re-requesting a page that remains visible; touch its LRU recency and skip the
            // (expensive, full material shading) re-render entirely.
            if (m_VTManager->IsPageResident(x, y, mip)) {
                m_VTManager->TouchPage(x, y, mip);
                m_RequestQueue.MarkRequestCompleted(key);
                continue;
            }

            // Allocates (evicting LRU if the physical pool is full -- see
            // VirtualTextureManager::AllocatePhysicalSlot for that algorithm) and writes this
            // frame's CPU-side page-table mirror entry, but does not touch any GPU image yet.
            uint32_t physicalPageIndex = m_VTManager->RequestPageResidency(cmd, x, y, mip);

            RenderPage(cmd, x, y, mip, physicalPageIndex, renderFn);
            renderedAnyPage = true;
            m_RequestQueue.MarkRequestCompleted(key);
        }

        if (renderedAnyPage) {
            // Flushes every PropagatePageTable() write RequestPageResidency made above (own CPU
            // mirror -> GPU page table image copy + barrier) in one batched call, rather than once
            // per page -- UpdatePageTableImage() early-outs entirely if nothing is actually dirty.
            m_VTManager->UpdatePageTableImage(cmd);
        }
    }

    void VirtualTextureRenderPass::RenderPage(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t mip,
        uint32_t physicalPageIndex, const VirtualTexturePageRenderFn& renderFn) {
        const uint32_t tileSizeWithBorder = m_VTManager->GetTileSizeWithBorder();
        const uint32_t poolCount = static_cast<uint32_t>(m_VTManager->GetPhysicalPoolCount());

        // =====================================================================================
        // STEP 1 -- Transition every physical pool's target layer from UNDEFINED (this call fully
        // overwrites the tile, so any previous contents are explicitly discarded, not preserved) to
        // COLOR_ATTACHMENT_OPTIMAL, ready for Dynamic Rendering to write into.
        // =====================================================================================
        std::vector<VkImageMemoryBarrier2> toAttachmentBarriers(poolCount);
        for (uint32_t i = 0; i < poolCount; ++i) {
            VkImageMemoryBarrier2& barrier = toAttachmentBarriers[i];
            barrier = VkImageMemoryBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_VTManager->GetPhysicalPoolImage(i);
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, physicalPageIndex, 1 };
        }
        VkDependencyInfo toAttachmentDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDep.imageMemoryBarrierCount = static_cast<uint32_t>(toAttachmentBarriers.size());
        toAttachmentDep.pImageMemoryBarriers = toAttachmentBarriers.data();
        vkCmdPipelineBarrier2(cmd, &toAttachmentDep);

        // =====================================================================================
        // STEP 2 -- Dynamic Rendering (VK_KHR_dynamic_rendering, no VkRenderPass/VkFramebuffer
        // overhead): one MRT color attachment per physical pool (e.g. Albedo, Normal, ORM), all
        // targeting the SAME physical page slot's own per-layer 2D view.
        // =====================================================================================
        std::vector<VkRenderingAttachmentInfo> colorAttachments(poolCount);
        for (uint32_t i = 0; i < poolCount; ++i) {
            colorAttachments[i] = VkRenderingAttachmentInfo{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            colorAttachments[i].imageView = m_VTManager->GetPhysicalPoolLayerView(i, physicalPageIndex);
            colorAttachments[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachments[i].clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        }

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, { tileSizeWithBorder, tileSizeWithBorder } };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
        renderingInfo.pColorAttachments = colorAttachments.data();
        vkCmdBeginRendering(cmd, &renderingInfo);

        // Virtual-viewport technique: see ComputePageViewport's own comment for the full derivation.
        VkViewport viewport{};
        ComputePageViewport(x, y, mip, viewport);
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        // Scissor clips rasterization to the physical tile's real (border-inclusive) extent --
        // without it, the oversized viewport above would let geometry outside this page's territory
        // rasterize into the attachment.
        VkRect2D scissor{ { 0, 0 }, { tileSizeWithBorder, tileSizeWithBorder } };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (renderFn) {
            maths::vec2 worldMin{};
            maths::vec2 worldMax{};
            ComputePageWorldRect(x, y, mip, worldMin, worldMax);
            renderFn(cmd, worldMin, worldMax, mip, m_VolumeViewProj);
        }

        vkCmdEndRendering(cmd);

        // =====================================================================================
        // STEP 3 -- Transition every physical pool's target layer from COLOR_ATTACHMENT_OPTIMAL to
        // SHADER_READ_ONLY_OPTIMAL: makes this page's freshly-written content visible to any later
        // fragment/compute shader sampled read THIS SAME frame (unlike VirtualShadowMapPass's
        // feedback-buffer contract, a page rendered here has no one-frame lag -- a consumer pass
        // later in this same frame's command buffer must already see the final result).
        // =====================================================================================
        std::vector<VkImageMemoryBarrier2> toShaderBarriers(poolCount);
        for (uint32_t i = 0; i < poolCount; ++i) {
            VkImageMemoryBarrier2& barrier = toShaderBarriers[i];
            barrier = VkImageMemoryBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_VTManager->GetPhysicalPoolImage(i);
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, physicalPageIndex, 1 };
        }
        VkDependencyInfo toShaderDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toShaderDep.imageMemoryBarrierCount = static_cast<uint32_t>(toShaderBarriers.size());
        toShaderDep.pImageMemoryBarriers = toShaderBarriers.data();
        vkCmdPipelineBarrier2(cmd, &toShaderDep);
    }

}
