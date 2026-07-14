#pragma once
#include <cstdint>
#include <atomic>
#include "core/maths/Maths.h"

namespace core {

    // Identifiant unique 64 bits
    using EntityID = uint64_t;

    class IDManager {
    public:
        // contextID : un identifiant unique par thread ou par "usine" (ex: 0, 1, 2...)
        // seed : ta SEED_GENERAL
        static void Init(uint16_t contextID, uint64_t seed = (uint64_t)maths::SEED_GENERAL) 
        {
            s_Context = static_cast<uint64_t>(contextID) << 48; // Décalage pour réserver les 16 bits de poids fort
            s_Sequence = 0;
            s_BaseSeed = seed;
        }

        // Renvoie un ID unique et déterministe
        static EntityID GetNextID() {
            // On combine le contexte fixe du thread et la séquence locale
            // C'est déterministe car GetNextID() appelé dans le même contexte
            // renverra toujours la même séquence 0, 1, 2...
            return s_Context | (s_Sequence++);
        }

        // Pour le GPU : récupère une graine "aléatoire" basée sur l'ID (sans état)
        static uint64_t GetRandomSeedFromID(EntityID id) {
            // On combine l'ID avec la seed générale pour créer une graine GPU unique
            return id ^ (s_BaseSeed * 0x9E3779B97F4A7C15ULL);
        }

    private:
        static inline uint64_t s_Context = 0;
        static inline uint64_t s_Sequence = 0;
        static inline uint64_t s_BaseSeed = 0;
    };
}