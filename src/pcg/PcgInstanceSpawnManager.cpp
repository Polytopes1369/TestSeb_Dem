// PCG framework roadmap, Phase 4.2 ("Spawner-to-DrawPass Glue"). See PcgInstanceSpawnManager.h for
// the full design rationale (ownership model, why this stays purely CPU-side bookkeeping, why pool
// exhaustion is an expected, not exceptional, condition).

#include "pcg/PcgInstanceSpawnManager.h"

#include "core/Logger.h"

#include <format>

namespace pcg {

    std::vector<uint32_t> PcgInstanceSpawnManager::SpawnInstances(const std::vector<PcgSpawnRequest>& requests) {
        std::vector<uint32_t> acquiredSlots;
        acquiredSlots.reserve(requests.size());

        uint32_t skippedCount = 0;
        for (const PcgSpawnRequest& request : requests) {
            // Field-order-matched call -- see PcgInstanceSpawnManager.h's own top-of-file comment
            // and PcgMeshSpawner.h's own PcgSpawnRequest declaration comment for why this is a
            // direct, no-reordering pass-through of every field.
            uint32_t slot = m_DrawPass.AcquireInstance(request.meshID, request.materialID,
                request.position, request.rotation, request.scale);
            if (slot == renderer::PcgInstanceDrawPass::kInvalidInstance) {
                // Expected, not exceptional -- see this file's header comment. Logged (Debug-only,
                // via core/Logger.h's own no-op-in-Release LOG_WARNING) so a real pool-sizing
                // shortfall is still visible during development, without ever crashing or aborting
                // the remaining requests in this same call.
                ++skippedCount;
                continue;
            }
            acquiredSlots.push_back(slot);
        }

        if (skippedCount > 0) {
            LOG_WARNING(std::format(
                "[PcgInstanceSpawnManager] SpawnInstances(): {} of {} request(s) could not be acquired "
                "(PcgInstanceDrawPass instance pool exhausted) -- skipped, not fatal.",
                skippedCount, requests.size()));
        }

        return acquiredSlots;
    }

    void PcgInstanceSpawnManager::DespawnInstances(const std::vector<uint32_t>& slots) {
        for (uint32_t slot : slots) {
            m_DrawPass.ReleaseInstance(slot);
        }
    }

}
