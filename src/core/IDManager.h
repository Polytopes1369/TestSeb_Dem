#pragma once
#include <cstdint>
#include <atomic>
#include "core/maths/Maths.h"

namespace core {

    using EntityID = uint64_t;

    class IDManager {
    public:
        static void Init(uint16_t contextID, uint64_t seed = (uint64_t)maths::SEED_GENERAL) 
        {
            s_Context = static_cast<uint64_t>(contextID) << 48; // Shift to reserve the 16 most significant bits
            s_Sequence = 0;
            s_BaseSeed = seed;
        }

        static EntityID GetNextID() {
            return s_Context | (s_Sequence++);
        }

        static uint64_t GetRandomSeedFromID(EntityID id) {
            return id ^ (s_BaseSeed * 0x9E3779B97F4A7C15ULL);
        }

    private:
        static inline uint64_t s_Context = 0;
        static inline uint64_t s_Sequence = 0;
        static inline uint64_t s_BaseSeed = 0;
    };
}
