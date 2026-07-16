#include "geometry/SurfaceCacheAtlasAllocator.h"

#include <algorithm>
#include <limits>

namespace geometry {

    SurfaceCacheAtlasAllocator::SurfaceCacheAtlasAllocator(uint32_t atlasSize)
        : m_AtlasSize(atlasSize) {
        Reset();
    }

    void SurfaceCacheAtlasAllocator::Reset() {
        m_FreeRects.clear();
        m_FreeRects.push_back(AtlasRect{ 0, 0, m_AtlasSize, m_AtlasSize });
    }

    bool SurfaceCacheAtlasAllocator::Allocate(uint32_t width, uint32_t height, AtlasRect& outRect) {
        if (width == 0 || height == 0) {
            return false;
        }

        // Best-area-fit: the smallest-area free rect that still fits (width, height) exactly --
        // minimizes the leftover area this one placement carves off, which in turn keeps the
        // guillotine splits below producing the largest possible leftover pieces for future
        // allocations (see class comment).
        size_t bestIndex = std::numeric_limits<size_t>::max();
        uint64_t bestArea = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < m_FreeRects.size(); ++i) {
            const AtlasRect& free = m_FreeRects[i];
            if (free.width < width || free.height < height) {
                continue;
            }
            const uint64_t area = static_cast<uint64_t>(free.width) * static_cast<uint64_t>(free.height);
            if (area < bestArea) {
                bestArea = area;
                bestIndex = i;
            }
        }

        if (bestIndex == std::numeric_limits<size_t>::max()) {
            return false;
        }

        const AtlasRect chosen = m_FreeRects[bestIndex];
        m_FreeRects.erase(m_FreeRects.begin() + static_cast<std::ptrdiff_t>(bestIndex));

        outRect = AtlasRect{ chosen.x, chosen.y, width, height };

        // Guillotine split, "shorter leftover axis" heuristic: the leftover L-shape (chosen minus
        // outRect) is cut into exactly two new free rects. Comparing the two leftover extents
        // decides which cut direction leaves one of the two new rects spanning a FULL original
        // edge (width or height) rather than both new rects being clipped on both axes.
        const uint32_t leftoverWidth = chosen.width - width;
        const uint32_t leftoverHeight = chosen.height - height;

        AtlasRect rightRect{};
        AtlasRect bottomRect{};
        if (leftoverWidth <= leftoverHeight) {
            // Bottom rect spans the chosen rect's full original width; right rect only spans the
            // placed height.
            bottomRect = AtlasRect{ chosen.x, chosen.y + height, chosen.width, leftoverHeight };
            rightRect = AtlasRect{ chosen.x + width, chosen.y, leftoverWidth, height };
        } else {
            // Right rect spans the chosen rect's full original height; bottom rect only spans the
            // placed width.
            rightRect = AtlasRect{ chosen.x + width, chosen.y, leftoverWidth, chosen.height };
            bottomRect = AtlasRect{ chosen.x, chosen.y + height, width, leftoverHeight };
        }

        if (rightRect.width > 0 && rightRect.height > 0) {
            m_FreeRects.push_back(rightRect);
        }
        if (bottomRect.width > 0 && bottomRect.height > 0) {
            m_FreeRects.push_back(bottomRect);
        }

        return true;
    }

    void SurfaceCacheAtlasAllocator::Free(const AtlasRect& rect) {
        if (rect.width == 0 || rect.height == 0) {
            return;
        }
        m_FreeRects.push_back(rect);
        CoalesceFreeRects();
    }

    void SurfaceCacheAtlasAllocator::CoalesceFreeRects() {
        // Repeat full O(n^2) merge passes until one finds no mergeable pair -- a single freed rect
        // can chain-merge with several neighbors in sequence (e.g. left neighbor, then the newly
        // widened rect merges with a rect above it), so a single pass is not always sufficient.
        bool mergedAny = true;
        while (mergedAny) {
            mergedAny = false;
            for (size_t i = 0; i < m_FreeRects.size() && !mergedAny; ++i) {
                for (size_t j = i + 1; j < m_FreeRects.size(); ++j) {
                    const AtlasRect& a = m_FreeRects[i];
                    const AtlasRect& b = m_FreeRects[j];

                    AtlasRect merged{};
                    bool canMerge = false;

                    // Horizontal merge: same row span (y, height), adjacent in x.
                    if (a.y == b.y && a.height == b.height) {
                        if (a.x + a.width == b.x) {
                            merged = AtlasRect{ a.x, a.y, a.width + b.width, a.height };
                            canMerge = true;
                        } else if (b.x + b.width == a.x) {
                            merged = AtlasRect{ b.x, b.y, a.width + b.width, a.height };
                            canMerge = true;
                        }
                    }
                    // Vertical merge: same column span (x, width), adjacent in y.
                    if (!canMerge && a.x == b.x && a.width == b.width) {
                        if (a.y + a.height == b.y) {
                            merged = AtlasRect{ a.x, a.y, a.width, a.height + b.height };
                            canMerge = true;
                        } else if (b.y + b.height == a.y) {
                            merged = AtlasRect{ b.x, b.y, a.width, a.height + b.height };
                            canMerge = true;
                        }
                    }

                    if (canMerge) {
                        // Erase j first (higher index) so i's position stays valid.
                        m_FreeRects.erase(m_FreeRects.begin() + static_cast<std::ptrdiff_t>(j));
                        m_FreeRects[i] = merged;
                        mergedAny = true;
                        break;
                    }
                }
            }
        }
    }

    uint64_t SurfaceCacheAtlasAllocator::FreeAreaTexels() const {
        uint64_t total = 0;
        for (const AtlasRect& rect : m_FreeRects) {
            total += static_cast<uint64_t>(rect.width) * static_cast<uint64_t>(rect.height);
        }
        return total;
    }

}
