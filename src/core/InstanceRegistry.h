#pragma once
// Phase 0.1 (UE5.8-parity PCG roadmap, "Dynamic Instance Registry" -- the foundational subtask the
// whole 10-phase PCG integration builds on): generalizes the engine's previous pattern of a fixed
// std::array<core::EntityData, N> ("the whole scene graph" -- every render pass indexes entities by
// meshID/materialID/flags read out of exactly this array) plus an ad-hoc, hand-rolled free-list of
// spare indices (see renderer::VulkanContext's own kStreamingUnitCount pool and main.cpp's
// freeStreamingUnits/cellToStreamingUnit bookkeeping around its WorldCellStreamingLoader consumer)
// into one reusable, capacity-bounded slot allocator.
//
// Design constraints (see this class's own call site in renderer::VulkanContext for the concrete
// instantiation):
//   - The engine's GPU SSBO upload path (renderer::VulkanContext::UploadEntityData/
//     PatchStreamingUnitEntityData) and every downstream consumer (culling, LOD selection, shading,
//     TLAS refit, Surface Cache, Global SDF) all assume a CONTIGUOUS, fixed-capacity backing array
//     they can either upload wholesale or index into directly by absolute slot number -- there is no
//     unbounded/resizable container here, by design, to stay compatible with that path without
//     touching a single one of those consumers. Capacity is a compile-time template parameter
//     (grow-on-demand UP TO that ceiling, never beyond it), not a runtime-resizable std::vector.
//   - AcquireSlot()/ReleaseSlot() use the exact same free-list idiom already established elsewhere
//     in this codebase for slot/index recycling under a hard capacity ceiling (renderer::
//     ParticleSystemPass's GPU-side CounterBuffer dead-list, and this project's own CPU-side
//     freeStreamingUnits precedent in main.cpp) -- a LIFO stack of released indices, checked before
//     ever bumping a monotonically-growing high-water mark into never-yet-used storage.
//   - NOT thread-safe: exactly like renderer::VulkanContext::SetStreamingUnitState (which this class
//     is meant to eventually back), AcquireSlot()/ReleaseSlot() are main-thread-only. Callers that
//     need to claim/release slots from a worker thread must stage the request into their own
//     thread-safe queue and apply it on the main thread, same as world::WorldCellStreamingLoader
//     already does for its own StreamingPlacementEvent queue.
//
// Not yet wired to any runtime caller as of Phase 0.1 -- renderer::VulkanContext currently only ever
// calls AcquireSlot() once per slot, at startup, and never ReleaseSlot()s a live entity. Later PCG
// roadmap phases are expected to actually drive Acquire/Release at runtime as procedurally spawned
// content streams in/out.

#include "core/EntityData.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace core {

    // Capacity-bounded pool of core::EntityData + core::EntityTransformCPU slots, indexed by a
    // uint32_t handle stable for the lifetime of the acquisition (i.e. until the matching
    // ReleaseSlot() call). Backing storage is two parallel std::array<T, Capacity> members --
    // Data()/TransformData() expose them as plain contiguous pointers with EXACTLY the same layout
    // and lifetime guarantees the pre-Phase-0.1 std::array<EntityData, kTotalEntityCount> member
    // gave every existing GPU-upload / "iterate every entity" consumer, so callers that only ever
    // read via those pointers need no changes at all.
    template <uint32_t Capacity>
    class InstanceRegistry {
    public:
        static constexpr uint32_t kCapacity = Capacity;
        // Returned by AcquireSlot() when the pool is fully exhausted (free list empty AND the
        // high-water mark has already reached Capacity) -- callers MUST check for this exactly like
        // main.cpp's own existing "Pool exhausted" check on its hand-rolled freeStreamingUnits stack.
        static constexpr uint32_t kInvalidSlot = 0xFFFFFFFFu;

        InstanceRegistry() {
            m_FreeList.reserve(Capacity);
        }

        // Claims one slot for a new instance, copying `initial` into it and resetting its transform
        // mirror to a default (identity rotation, zero center/translation -- see
        // core::EntityTransformCPU's own comment) -- the caller is responsible for writing a real
        // transform afterward via Transform(index), exactly like renderer::VulkanContext's
        // UpdateEntityRotations() already does once per frame for every currently-acquired index.
        //
        // Reuses the most-recently-released index first (LIFO free list -- see this class's own
        // header comment for why), falling back to bumping the high-water mark into storage that has
        // never been handed out before. Returns kInvalidSlot if the pool is completely exhausted;
        // never throws and never blocks.
        uint32_t AcquireSlot(const EntityData& initial) {
            uint32_t index;
            if (!m_FreeList.empty()) {
                index = m_FreeList.back();
                m_FreeList.pop_back();
            } else if (m_HighWaterMark < Capacity) {
                index = m_HighWaterMark++;
            } else {
                return kInvalidSlot;
            }

            // A bug in the free-list/high-water-mark bookkeeping above would otherwise silently hand
            // out an index that is already live, double-writing over another owner's slot -- this
            // assert (compiled out in Release, same NDEBUG gate every other invariant check in this
            // engine uses) turns that into an immediate, loud failure instead of quiet corruption.
            assert(!m_Occupied[index] && "InstanceRegistry::AcquireSlot handed out an already-occupied index");
            m_Occupied[index] = true;
            m_Slots[index] = initial;
            m_Transforms[index] = EntityTransformCPU{};
            ++m_LiveCount;
            return index;
        }

        // Releases a previously-acquired slot back to the free list for future reuse. Resets its
        // EntityData/EntityTransformCPU to a fully inert default (zeroed flags/meshID/materialID,
        // identity transform) so a stray read of a released-but-not-yet-reacquired slot can never be
        // mistaken for live content.
        //
        // IMPORTANT for callers migrating existing "parked but still valid" slots (e.g. world::
        // WorldCellStreamingLoader's pre-baked streaming archetypes, which must always keep a real,
        // valid meshID even while flagged core::EntityFlags::StreamingInactive so their baked Nanite
        // cluster DAG never desyncs from re-written vertex data -- see that flag's own comment): such
        // slots must NOT be routed through ReleaseSlot(). They stay permanently acquired for the
        // registry's lifetime and manage their own "inactive" appearance purely via EntityData flags,
        // exactly as renderer::VulkanContext::SetStreamingUnitState already does today.
        void ReleaseSlot(uint32_t index) {
            assert(index < m_HighWaterMark && "InstanceRegistry::ReleaseSlot index was never acquired");
            assert(m_Occupied[index] && "InstanceRegistry::ReleaseSlot double-release (or release of a never-acquired slot)");
            m_Occupied[index] = false;
            m_Slots[index] = EntityData{};
            m_Transforms[index] = EntityTransformCPU{};
            m_FreeList.push_back(index);
            --m_LiveCount;
        }

        bool IsSlotOccupied(uint32_t index) const {
            assert(index < Capacity);
            return m_Occupied[index];
        }

        // Number of slots currently acquired (== high-water mark minus however many are on the free
        // list right now).
        uint32_t GetLiveCount() const { return m_LiveCount; }

        // Next never-yet-handed-out index -- only ever grows, even across ReleaseSlot() calls (a
        // released index goes back on the free list, not "below" the high-water mark conceptually).
        uint32_t GetHighWaterMark() const { return m_HighWaterMark; }

        static constexpr uint32_t GetCapacity() { return Capacity; }

        // --- Contiguous raw access, for the GPU upload path and every existing "iterate every
        // entity up to some fixed count" consumer -- identical semantics to the raw std::array this
        // class replaces. Callers must only ever read/iterate indices they know to be occupied (or
        // rely on a separately-tracked count, e.g. renderer::VulkanContext::GetEntityCount()) --
        // this class does not itself filter released/never-acquired slots out of Data()'s range. ---
        EntityData* Data() { return m_Slots.data(); }
        const EntityData* Data() const { return m_Slots.data(); }

        EntityData& operator[](uint32_t index) {
            assert(index < Capacity);
            return m_Slots[index];
        }
        const EntityData& operator[](uint32_t index) const {
            assert(index < Capacity);
            return m_Slots[index];
        }

        EntityTransformCPU* TransformData() { return m_Transforms.data(); }
        const EntityTransformCPU* TransformData() const { return m_Transforms.data(); }

        EntityTransformCPU& Transform(uint32_t index) {
            assert(index < Capacity);
            return m_Transforms[index];
        }
        const EntityTransformCPU& Transform(uint32_t index) const {
            assert(index < Capacity);
            return m_Transforms[index];
        }

    private:
        std::array<EntityData, Capacity> m_Slots{};
        std::array<EntityTransformCPU, Capacity> m_Transforms{};
        // Occupancy bitset used purely for the AcquireSlot()/ReleaseSlot() asserts above -- tiny
        // (Capacity bytes) and unconditionally present in both Debug and Release, same as the rest
        // of this class: it is production bookkeeping (double-free/use-after-release detection via
        // <cassert>, which already compiles to nothing under NDEBUG on its own), not the kind of
        // debug-only diagnostic tooling CLAUDE.md requires to be compiled out of Release entirely.
        std::array<bool, Capacity> m_Occupied{};
        std::vector<uint32_t> m_FreeList; // LIFO stack of released indices.
        uint32_t m_HighWaterMark = 0;
        uint32_t m_LiveCount = 0;
    };

}
