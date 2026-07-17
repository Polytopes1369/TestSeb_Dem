#pragma once
// 128-bit identifier used to name every World Partition actor. An actor's UUID doubles as its
// "One File Per Actor" filename stem (see OfpaActor.h::MakeActorFilePath), so identity survives
// file moves, renames and re-parenting exactly the way UE5.8's per-actor FGuid scheme does.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>

namespace worldpartition {

    struct Uuid {
        uint64_t high = 0;
        uint64_t low = 0;

        constexpr bool operator==(const Uuid& other) const { return high == other.high && low == other.low; }
        constexpr bool operator!=(const Uuid& other) const { return !(*this == other); }
        constexpr bool operator<(const Uuid& other) const {
            return high != other.high ? high < other.high : low < other.low;
        }

        constexpr bool IsNil() const { return high == 0 && low == 0; }

        // 32-character lowercase hex string (e.g. "3fa85f6457174562b3fc2c963f66afa") -- also this
        // actor's ".actor" filename stem, see OfpaActor.h::MakeActorFilePath.
        std::string ToHexString() const;
    };

    inline constexpr Uuid kNilUuid{};

    // RFC 4122 version-4 (random) UUID generator: stamps the version nibble to 4 and the variant
    // bits to RFC 4122's "10", every other bit drawn from a PRNG. Adequate for offline
    // content-authoring identity (collision probability is astronomically low at any realistic
    // actor count) -- NOT intended for anything security-sensitive.
    class UuidGenerator {
    public:
        // Seeds from std::random_device (OS entropy) -- the normal, non-deterministic path used
        // by actual content-authoring tools.
        UuidGenerator();

        // Deterministic seed, for reproducible test fixtures (see tests/OfpaSerializationTests.cpp).
        explicit UuidGenerator(uint64_t seed);

        Uuid Generate();

    private:
        std::mt19937_64 engine_;
        std::uniform_int_distribution<uint64_t> dist_{ 0, UINT64_MAX };
    };

}

namespace std {
    template<>
    struct hash<worldpartition::Uuid> {
        size_t operator()(const worldpartition::Uuid& id) const noexcept {
            // 64-bit hash-combine (boost::hash_combine's constant): folds `low`'s bits into a
            // hash seeded from `high` so both halves contribute even into a power-of-two-sized
            // unordered_map bucket count (a plain XOR would let either half's high bits, which a
            // pow2 bucket count discards, go untouched).
            size_t h = std::hash<uint64_t>{}(id.high);
            h ^= std::hash<uint64_t>{}(id.low) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
}
