#pragma once
// Phase: Virtual Texturing Step 3 (Runtime Virtual Texturing -- on-demand GPU tile rendering).
//
// This pass renders a virtual texture's PHYSICAL PAGES on demand, directly into
// renderer::VirtualTextureManager's physical pool atlas, instead of the CPU-upload path
// (VirtualTextureManager::UploadTileData) Step 1 already provides for pre-baked/streamed source
// data. It is the RVT half of SVT/RVT: terrain material blending, decal projection, or any other
// "bake it once per page, sample it many times" content whose source is GEOMETRY (drawn with a
// pipeline) rather than a texture already sitting on disk.
//
// --- Why a caller-supplied render callback, unlike VirtualShadowMapPass's fixed pipeline ---
// renderer::VirtualShadowMapPass owns one fixed depth-only pipeline because every VSM page draws
// the exact same fallback-mesh geometry with the exact same shader. A virtual texture has no such
// uniform content: a terrain page and a decal page need different geometry AND different material
// shaders (albedo vs. normal vs. ORM, or entirely different pipelines per RVT layer in UE5.8 terms).
// This class therefore owns none of the geometry or the pipeline -- it owns exactly the boilerplate
// that IS common to every page (dynamic-rendering setup, the virtual-viewport
// technique, image layout barriers, page-table residency + flush) and hands control back to the
// caller via VirtualTexturePageRenderFn for the pipeline-specific bind/push-constant/draw calls.
//
// --- The virtual-viewport technique (borrowed from VirtualShadowMapPass::RenderPage) ---
// Rather than deriving a per-page ASYMMETRIC/oblique projection matrix, this class keeps ONE fixed,
// symmetric top-down orthographic projection covering the WHOLE virtual texture volume (computed
// once at Init(), see m_VolumeViewProj) and, for each page, offsets an OVERSIZED viewport so only
// that page's own sub-rectangle of the fixed projection lands inside the physical tile's bounds --
// Vulkan explicitly permits a viewport larger than the bound attachment (see
// VirtualShadowMapPool.h's own comment on why this is guaranteed safe on any conformant Vulkan 1.3
// device without a capability query); the scissor clips rasterization to the tile's real extent.
// The viewport is sized in units of GetTileSizeWithBorder() (not the bare tile size): because the
// fixed projection is continuous across page boundaries, a viewport pixel that lands just past one
// page's content area is exactly the neighboring page's content -- which is precisely what a tile's
// border texels need to sample for seam-free bilinear filtering (see virtual_texture_lookup.glsl's
// own border-bias math). A coarser mip level's page covers a LARGER slice of the SAME fixed world
// extent, so only the viewport's pixel-density (its width/height, computed from that mip's page
// grid resolution) changes per mip -- the projection itself never does.
//
// --- Per-frame call order a caller must follow ---
//   1. RequestPage(x, y, mip) -- called any number of times (e.g. from a terrain streaming system's
//      visibility pass, or a shader's own feedback report resolved on the CPU) to mark a page as
//      needed. Deduplicated internally via geometry::StreamingRequestQueue, exactly like
//      VirtualShadowMapPass's own request-queue contract.
//   2. RecordPageUpdates(cmd, renderFn) -- once per frame: drains up to kMaxPagesRenderedPerFrame
//      queued requests, resolves physical residency (allocating + evicting LRU pages under the hood
//      via VirtualTextureManager::RequestPageResidency -- see that class for the CPU-side LRU
//      eviction algorithm this class relies on, not reimplements), renders each via `renderFn`
//      between the required color-attachment barriers, and flushes the CPU page-table mirror to the
//      GPU-resident page table image (VirtualTextureManager::UpdatePageTableImage) so this frame's
//      newly-rendered pages are visible to any pass sampling the virtual texture afterward.

#include <cstdint>
#include <functional>
#include <vector>
#include <vulkan/vulkan.h>

#include "core/maths/Maths.h"
#include "io/StreamingRequestQueue.h"
#include "renderer/streaming/VirtualTextureManager.h"

namespace renderer {

    // World-space volume this virtual texture's [0,1]^2 UV space maps onto: a top-down (Y-up)
    // projection over the XZ plane, matching Unreal Engine's URuntimeVirtualTextureVolume
    // convention. heightMin/heightMax bound the ortho near/far planes along Y so geometry above or
    // below the volume is clipped, not just off to the side.
    struct VirtualTextureVolumeBounds {
        maths::vec2 worldMinXZ{ -8.0f, -8.0f };
        maths::vec2 worldMaxXZ{ 8.0f, 8.0f };
        float heightMin = -4.0f;
        float heightMax = 4.0f;
    };

    // Callback the caller provides to actually draw a page's content. `worldRectMin`/`worldRectMax`
    // (XZ, already expanded to include this page's border coverage -- see class comment) is what the
    // callback should use to cull/isolate which terrain patches or static objects overlap this page
    // (the "isolate the portions of geometry" requirement); `viewProj` is the exact matrix that will
    // rasterize correctly under the viewport/scissor this class has already bound -- the callback
    // MUST use this matrix verbatim (not derive its own from worldRectMin/Max) for the virtual-
    // viewport technique to line up. Called once per resident color format set (i.e. once per page,
    // with every one of GetPhysicalPoolCount() attachments bound as MRT) inside an active
    // vkCmdBeginRendering/vkCmdEndRendering scope -- the callback only binds pipeline + pushes
    // constants/descriptors + draws, it must not begin/end rendering itself.
    using VirtualTexturePageRenderFn = std::function<void(
        VkCommandBuffer cmd,
        const maths::vec2& worldRectMin,
        const maths::vec2& worldRectMax,
        uint32_t mip,
        const maths::mat4& viewProj)>;

    class VirtualTextureRenderPass {
        friend class VirtualTextureRenderPassTests;
    public:
        VirtualTextureRenderPass() = default;

        VirtualTextureRenderPass(const VirtualTextureRenderPass&) = delete;
        VirtualTextureRenderPass& operator=(const VirtualTextureRenderPass&) = delete;

        // Bounded per-frame render budget -- mirrors VirtualShadowMapPass::kMaxPagesRenderedPerFrame's
        // own rationale (avoid one frame's queued backlog stalling the GPU with an unbounded number
        // of dynamic-rendering begin/end scopes). Tile content here is typically far more expensive
        // than VSM's depth-only draw (full material shading), hence the much smaller default.
        static inline uint32_t kMaxPagesRenderedPerFrame = 64u;

        // `vtManager` must outlive this pass and have already been Init()'d -- not owned here (this
        // class only adds the RENDERING half on top of VirtualTextureManager's already-complete
        // page-table + physical-pool bookkeeping from Step 1).
        bool Init(VkDevice device, VirtualTextureManager* vtManager, const VirtualTextureVolumeBounds& bounds);

        void Shutdown();

        // Marks (x, y, mip) as needed; safe to call multiple times per frame for the same page
        // (deduplicated, see class comment).
        void RequestPage(uint32_t x, uint32_t y, uint32_t mip);

        // Drains and renders up to kMaxPagesRenderedPerFrame queued requests. No-op (does not touch
        // the page table image) if nothing was rendered this call, mirroring
        // VirtualTextureManager::UpdatePageTableImage's own early-out on a clean CPU mirror.
        void RecordPageUpdates(VkCommandBuffer cmd, const VirtualTexturePageRenderFn& renderFn);

        // Pure-CPU helpers (no Vulkan calls) -- exposed for VirtualTextureRenderPassTests and for
        // callers that want to precompute a page's world rect without going through RecordPageUpdates
        // (e.g. a terrain LOD selector deciding which pages are visible enough to RequestPage()).

        // World-space rectangle (XZ) this page covers, expanded on every side by the border's world-
        // space footprint at this mip -- see class comment on why this exact expansion is what makes
        // border texels sample correct neighboring content.
        void ComputePageWorldRect(uint32_t x, uint32_t y, uint32_t mip, maths::vec2& outMin, maths::vec2& outMax) const;

        // Virtual-viewport parameters for a page: the oversized viewport offset/extent (in tile-with-
        // border-sized units) that must be bound alongside the fixed GetVolumeViewProj() projection
        // for that page's content to land correctly inside its physical tile.
        void ComputePageViewport(uint32_t x, uint32_t y, uint32_t mip, VkViewport& outViewport) const;

        const maths::mat4& GetVolumeViewProj() const { return m_VolumeViewProj; }

        size_t PendingRequestCount() const { return m_RequestQueue.PendingCount(); }

    private:
        void RenderPage(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t mip, uint32_t physicalPageIndex,
            const VirtualTexturePageRenderFn& renderFn);

        VkDevice m_Device = VK_NULL_HANDLE;
        VirtualTextureManager* m_VTManager = nullptr;
        VirtualTextureVolumeBounds m_Bounds{};

        // Fixed for this pass' entire lifetime -- see class comment on the virtual-viewport
        // technique for why the projection never needs to vary per page or per mip.
        maths::mat4 m_VolumeViewProj{};

        geometry::StreamingRequestQueue m_RequestQueue; // Keyed by VirtualTextureManager::PackPageKey.
        // Batches this frame's RequestPage() calls for a single StreamingRequestQueue::
        // SubmitFrameRequests() call in RecordPageUpdates(), instead of one per RequestPage() call --
        // matches that class' own "fold each frame's reports in one batch" design intent (see its
        // header comment) and avoids one small heap allocation per requested page.
        std::vector<uint32_t> m_PendingThisFrameKeys;
    };

}
