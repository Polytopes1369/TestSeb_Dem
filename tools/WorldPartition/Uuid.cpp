#include "Uuid.h"

#include <iomanip>
#include <sstream>

namespace worldpartition {

    std::string Uuid::ToHexString() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(16) << high << std::setw(16) << low;
        return oss.str();
    }

    UuidGenerator::UuidGenerator() {
        std::random_device rd;
        std::seed_seq seed{ rd(), rd(), rd(), rd() };
        engine_.seed(seed);
    }

    UuidGenerator::UuidGenerator(uint64_t seed) : engine_(seed) {}

    Uuid UuidGenerator::Generate() {
        uint64_t hi = dist_(engine_);
        uint64_t lo = dist_(engine_);

        // Stamp RFC 4122 version 4 into bits 12-15 of the high 64 bits (the "time_hi_and_version"
        // field's top nibble) and variant "10" into the top two bits of the low 64 bits (the
        // "clock_seq_hi_and_reserved" byte), matching the canonical
        // xxxxxxxx-xxxx-4xxx-{8,9,A,B}xxx-xxxxxxxxxxxx layout.
        hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
        lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

        return Uuid{ hi, lo };
    }

}
