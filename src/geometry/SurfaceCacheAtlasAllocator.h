#pragma once
// Runtime 2D page allocator for the Lumen-style Surface Cache atlas (see renderer::SurfaceCachePass).
//
// Unlike CardGenerator::PackCardsIntoAtlas (a ONE-SHOT, offline, whole-scene shelf pack baked into
// the .cache file at build time -- see CardGenerator.h's own two-phase API comment), this is a
// live, mutable allocator: renderer::SurfaceCachePass calls Allocate() when a card becomes visible
// on screen (residency requested) and Free() when it has been off-screen for long enough to evict
// (residency revoked), so the atlas only ever holds pages for cards actually worth shading THIS
// session, and a card leaving the screen releases its texels for another card entering it -- the
// same "streaming virtual texture" idea renderer::GlobalSDFPass's own class comment already
// documents for its clipmap volumes, applied here to a 2D atlas of independently-sized rects
// instead of a fixed-size voxel window.
//
// --- Algorithm: Guillotine bin packing with free-rectangle merging ---
// The allocator owns a list of disjoint, atlas-space free rectangles (initially one rect covering
// the whole atlas). Allocate(w, h):
//   1. Best-area-fit: scan every free rect that is >= (w, h) in both dimensions, keep the one with
//      the smallest AREA (minimizes wasted space left behind by this particular placement).
//   2. Guillotine split ("shorter leftover axis" rule): the chosen free rect minus the placed (w,
//      h) region leaves an L-shaped remainder; split that scan-line-fashion into exactly two new
//      free rects (never more), choosing the horizontal-vs-vertical cut so whichever leftover
//      dimension (width or height) is SMALLER ends up as an edge of one full-length new rect
//      rather than being sliced twice -- the standard guillotine heuristic for keeping leftover
//      slivers usable instead of shrinking them further on a later split.
// Free(rect) reinserts `rect` into the free list, then repeatedly coalesces any two free rects that
// share a full edge (same y + height, adjacent x -- or same x + width, adjacent y) into one larger
// rect, so pages freed next to each other recombine into placements a later Allocate() can actually
// use, instead of leaving the free list permanently shredded into the exact shapes prior
// allocations happened to vacate.
//
// --- Fragmentation ---
// Guillotine packing (even with merging) can still fail an Allocate() call the atlas has
// theoretically enough FreeAreaTexels() for, if that area is scattered across rects each smaller
// than (w, h) individually -- classic external fragmentation. FreeAreaTexels() lets a caller
// distinguish that ("plenty of free area, just fragmented -- worth evicting more and/or rebuilding
// from scratch via Reset() + replaying every still-wanted allocation") from genuine
// over-subscription (not enough free area at all, no amount of defragmentation would help).
// renderer::SurfaceCachePass is the caller that makes this decision.

#include <cstdint>
#include <vector>

namespace geometry {

    // An axis-aligned rectangle of atlas texels, top-left origin (matches SurfaceCacheCardEntry's
    // own atlasOffset/atlasSize convention).
    struct AtlasRect {
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    class SurfaceCacheAtlasAllocator {
    public:
        // Starts with a single free rect covering the whole [0, atlasSize) x [0, atlasSize) square.
        explicit SurfaceCacheAtlasAllocator(uint32_t atlasSize);

        // Discards every allocation and free-list entry, resetting to one free rect covering the
        // whole atlas again -- used by a caller performing a full defragmentation rebuild (Free()
        // every currently-resident rect logically, Reset(), then Allocate() them again in a fresh,
        // tightly-packed order).
        void Reset();

        // Best-area-fit + guillotine split (see class comment). Returns false (outRect untouched)
        // if no single free rect is large enough to hold (width, height) -- the caller decides
        // whether that means "evict more" or "atlas exhausted," using FreeAreaTexels() as a guide.
        bool Allocate(uint32_t width, uint32_t height, AtlasRect& outRect);

        // Returns `rect` to the free list and coalesces it with any adjacent free rects (see class
        // comment). `rect` must be exactly a rect previously returned by Allocate() (or the exact
        // union of several such rects) -- freeing a sub-region of a prior allocation is not
        // supported (this allocator has no notion of splitting a live allocation).
        void Free(const AtlasRect& rect);

        // Sum of every free rect's area, in texels -- a fragmentation diagnostic (see class
        // comment), not by itself a guarantee any single future Allocate() call will succeed.
        uint64_t FreeAreaTexels() const;

        size_t FreeRectCount() const { return m_FreeRects.size(); }
        uint32_t AtlasSize() const { return m_AtlasSize; }

    private:
        // Repeatedly scans for a mergeable pair (shared full edge, see class comment) and merges
        // it, until a full pass finds none -- O(n^2) per Free() call, acceptable because this only
        // runs on eviction (a handful of times per frame at most), never per-frame on every
        // resident card.
        void CoalesceFreeRects();

        uint32_t m_AtlasSize;
        std::vector<AtlasRect> m_FreeRects;
    };

}
