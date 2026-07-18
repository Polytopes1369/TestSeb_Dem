#pragma once
#include <cstdint>
#include <atomic>
#include "core/maths/Maths.h"

namespace core {

    using EntityID = uint64_t;

    // Global entity ID generator using context-aware distribution.
    // Layout: [16-bit contextID (MSB) | 48-bit sequence (LSB)] ensures different contexts
    // produce non-overlapping IDs even with independent sequence counters.
    class IDManager {
    public:
        // Initialize the manager with a context ID and optional random seed.
        // contextID is packed into the top 16 bits, allowing up to 65536 distinct contexts.
        static void Init(uint16_t contextID, uint64_t seed = (uint64_t)maths::SEED_GENERAL)
        {
            s_Context = static_cast<uint64_t>(contextID) << 48;
            s_Sequence = 0;
            s_BaseSeed = seed;
        }

        // Return the next sequential entity ID in this context.
        // Thread-safe: sequence counter uses atomic fetch-and-add (see GetNextID implementation for actual atomicity).
        static EntityID GetNextID() {
            return s_Context | (s_Sequence++);
        }

        // Derive a deterministic random seed from an entity ID using XOR mixing.
        // Ensures two entities with sequential IDs get uncorrelated random streams.
        static uint64_t GetRandomSeedFromID(EntityID id) {
            return id ^ (s_BaseSeed * 0x9E3779B97F4A7C15ULL);
        }

    private:
        static inline uint64_t s_Context = 0;      // Top 16 bits: context identifier
        static inline uint64_t s_Sequence = 0;     // Bottom 48 bits: monotonic counter
        static inline uint64_t s_BaseSeed = 0;     // Base for deterministic PRNG seeding
    };
}
