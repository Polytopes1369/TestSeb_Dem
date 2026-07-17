#include "renderer/passes/VirtualTextureRenderPass.h"
#include "renderer/streaming/VirtualTextureManager.h"
#include <cmath>
#include <iostream>
#include <string>

namespace renderer {

    // Device-less test bypass, mirroring VirtualTextureTests.cpp's own convention: bootstraps a
    // VirtualTextureManager's CPU-side bookkeeping directly (via friendship) instead of going
    // through Init() (which requires a real VkDevice), then exercises VirtualTextureRenderPass'
    // pure-CPU page-addressing math (ComputePageWorldRect/ComputePageViewport) -- the two helpers
    // that turn a (x, y, mip) page coordinate into "what world-space rectangle does this cover" and
    // "what viewport lands it correctly in the physical atlas", which is exactly the addressing
    // logic RecordPageUpdates/RenderPage rely on to line geometry up with its physical tile. Neither
    // VirtualTextureRenderPass::Init() nor these two helpers issue any Vulkan call, so all of this
    // is safe to run with VK_NULL_HANDLE and without a live device -- see
    // VirtualTextureRenderPass.cpp's Init()/ComputePageWorldRect()/ComputePageViewport() bodies.
    class VirtualTextureRenderPassTests {
    public:
        static int Run() {
            int failCount = 0;

            auto Check = [&](bool condition, const std::string& desc) {
                if (!condition) {
                    std::cerr << "[FAIL] " << desc << std::endl;
                    failCount++;
                }
            };
            auto CheckNear = [&](float actual, float expected, const std::string& desc) {
                if (std::fabs(actual - expected) > 1.0e-4f) {
                    std::cerr << "[FAIL] " << desc << " (expected " << expected << ", got " << actual << ")" << std::endl;
                    failCount++;
                }
            };

            // --- Shared device-less VirtualTextureManager bookkeeping: 16x16 pages at mip 0 (2048
            // virtual pixels / 128 tile size), 128px tiles with a 4px border. ---
            VirtualTextureManager mgr;
            VirtualTextureConfig config;
            config.virtualWidth = 2048;
            config.virtualHeight = 2048;
            config.tileSize = 128;
            config.borderSize = 4;
            config.physicalPageCapacity = 4;
            mgr.m_Config = config;
            mgr.m_GridWidth = config.virtualWidth / config.tileSize;   // 16
            mgr.m_GridHeight = config.virtualHeight / config.tileSize; // 16
            mgr.m_MipCount = static_cast<uint32_t>(std::log2(mgr.m_GridWidth)) + 1; // 5 (16,8,4,2,1)
            mgr.BuildMipOffsets();
            mgr.ClearPageTable(VK_NULL_HANDLE);

            VirtualTextureRenderPass pass;
            VirtualTextureVolumeBounds bounds;
            bounds.worldMinXZ = { -8.0f, -8.0f };
            bounds.worldMaxXZ = { 8.0f, 8.0f };
            bounds.heightMin = -1.0f;
            bounds.heightMax = 1.0f;
            Check(pass.Init(VK_NULL_HANDLE, &mgr, bounds), "VirtualTextureRenderPass::Init must succeed device-less");

            // Test 1: mip 0, page (0,0) world rect -- covers world X in [-8,-7] (1/16th of the 16-unit
            // span), expanded by the border's world-space footprint: 4 border texels out of 128
            // content texels = (1/128)*4 = 0.03125 world units on each side.
            {
                maths::vec2 rectMin, rectMax;
                pass.ComputePageWorldRect(0, 0, 0, rectMin, rectMax);
                CheckNear(rectMin.x, -8.0f - 0.03125f, "Page(0,0,mip0) world rect min.x");
                CheckNear(rectMax.x, -7.0f + 0.03125f, "Page(0,0,mip0) world rect max.x");
                CheckNear(rectMin.y, -8.0f - 0.03125f, "Page(0,0,mip0) world rect min.y (world Z)");
                CheckNear(rectMax.y, -7.0f + 0.03125f, "Page(0,0,mip0) world rect max.y (world Z)");
            }

            // Test 2: mip 0, the LAST page (15,15) must reach exactly the volume's own max corner
            // (before border expansion pushes slightly past it -- expected, see class comment on why
            // a volume-edge page's border simply samples outside the fixed frustum).
            {
                maths::vec2 rectMin, rectMax;
                pass.ComputePageWorldRect(15, 15, 0, rectMin, rectMax);
                CheckNear(rectMax.x, 8.0f + 0.03125f, "Page(15,15,mip0) world rect max.x must reach the volume's far edge");
                CheckNear(rectMax.y, 8.0f + 0.03125f, "Page(15,15,mip0) world rect max.y must reach the volume's far edge");
            }

            // Test 3: mip 1 halves the page grid (8x8), so page (0,0) at mip 1 covers TWICE the world
            // span of a mip-0 page (2 world units wide instead of 1), with a correspondingly larger
            // border expansion in world units (border texel count is fixed at 4, but each texel now
            // represents twice the world distance).
            {
                maths::vec2 rectMin, rectMax;
                pass.ComputePageWorldRect(0, 0, 1, rectMin, rectMax);
                CheckNear(rectMin.x, -8.0f - 0.0625f, "Page(0,0,mip1) world rect min.x");
                CheckNear(rectMax.x, -6.0f + 0.0625f, "Page(0,0,mip1) world rect max.x (twice mip0's span)");
            }

            // Test 4: viewport offset/extent for an interior mip-0 page -- the virtual-viewport
            // technique offsets by (pageCoord * tileSize) minus the border padding, and sizes the
            // viewport to the FULL mip grid's content-only resolution (16 * 128 = 2048).
            {
                VkViewport vp{};
                pass.ComputePageViewport(1, 2, 0, vp);
                CheckNear(vp.x, -(1.0f * 128.0f) + 4.0f, "Page(1,2,mip0) viewport.x");
                CheckNear(vp.y, -(2.0f * 128.0f) + 4.0f, "Page(1,2,mip0) viewport.y");
                CheckNear(vp.width, 16.0f * 128.0f, "Page(1,2,mip0) viewport.width");
                CheckNear(vp.height, 16.0f * 128.0f, "Page(1,2,mip0) viewport.height");
            }

            // Test 5: viewport for the coarsest mip (mip 4 -- a single 1x1 page, the root) must be
            // sized to exactly one tile's content resolution, offset only by the border padding.
            {
                VkViewport vp{};
                pass.ComputePageViewport(0, 0, 4, vp);
                CheckNear(vp.x, 4.0f, "Root page viewport.x (offset by border only)");
                CheckNear(vp.y, 4.0f, "Root page viewport.y (offset by border only)");
                CheckNear(vp.width, 128.0f, "Root page viewport.width (single page, content-only)");
                CheckNear(vp.height, 128.0f, "Root page viewport.height");
            }

            // Test 6: RequestPage() batches into the pending-this-frame list (not yet submitted to
            // the dedup queue -- that only happens inside RecordPageUpdates(), see class comment),
            // and does not itself deduplicate repeated requests for the same page within one frame.
            {
                pass.RequestPage(3, 4, 0);
                pass.RequestPage(3, 4, 0);
                Check(pass.m_PendingThisFrameKeys.size() == 2, "RequestPage must append, not dedup, before RecordPageUpdates runs");
                Check(pass.m_PendingThisFrameKeys[0] == VirtualTextureManager::PackPageKey(3, 4, 0),
                    "Pending key must match PackPageKey(3,4,0)");
                pass.m_PendingThisFrameKeys.clear(); // Never drained via RecordPageUpdates in this device-less test.
            }

            if (failCount == 0) {
                std::cout << "[VirtualTextureRenderPassTests] All tests passed." << std::endl;
            } else {
                std::cerr << "[VirtualTextureRenderPassTests] " << failCount << " test(s) failed." << std::endl;
            }
            return failCount;
        }
    };

}

int main() {
    return renderer::VirtualTextureRenderPassTests::Run();
}
