#include "OfpaActor.h"

#include <array>
#include <fstream>

namespace worldpartition {

    namespace {

        // Rotates `v` by unit quaternion `q` (Rodrigues' formula, quaternion form):
        // v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
        // Kept local to this file rather than added to core/maths/Maths.h::quat: no other part of
        // the engine currently needs quat-vector rotation, and this offline tool is deliberately
        // isolated from the rest of the engine (matching io/CacheFileManager.h's own isolation
        // note) so it never grows an accidental runtime dependency.
        maths::vec3 RotateVectorByQuat(const maths::quat& q, const maths::vec3& v) {
            maths::vec3 qv{ q.x, q.y, q.z };
            maths::vec3 t = qv.Cross(v) + v * q.w;
            return v + qv.Cross(t) * 2.0f;
        }

        // ---- Primitive binary write/read helpers -------------------------------------------
        // Every read helper returns false (leaving `out` untouched or partially written) the
        // moment the stream stops being good, so callers can bail out of a multi-field record
        // the instant corruption/truncation is detected instead of reading garbage into later
        // fields.

        template<typename T>
        void WriteRaw(std::ofstream& out, const T& value) {
            out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        }

        template<typename T>
        bool ReadRaw(std::ifstream& in, T& out) {
            in.read(reinterpret_cast<char*>(&out), sizeof(T));
            return in.good();
        }

        void WriteString(std::ofstream& out, const std::string& s) {
            uint32_t len = static_cast<uint32_t>(s.size());
            WriteRaw(out, len);
            if (len > 0) {
                out.write(s.data(), len);
            }
        }

        bool ReadString(std::ifstream& in, std::string& out) {
            uint32_t len = 0;
            if (!ReadRaw(in, len)) return false;
            out.resize(len);
            if (len > 0) {
                in.read(out.data(), len);
            }
            return in.good();
        }

        void WriteUuid(std::ofstream& out, const Uuid& id) {
            WriteRaw(out, id.high);
            WriteRaw(out, id.low);
        }

        bool ReadUuid(std::ifstream& in, Uuid& out) {
            return ReadRaw(in, out.high) && ReadRaw(in, out.low);
        }

        void WriteVec3(std::ofstream& out, const maths::vec3& v) {
            WriteRaw(out, v.x); WriteRaw(out, v.y); WriteRaw(out, v.z);
        }

        bool ReadVec3(std::ifstream& in, maths::vec3& v) {
            return ReadRaw(in, v.x) && ReadRaw(in, v.y) && ReadRaw(in, v.z);
        }

        void WriteQuat(std::ofstream& out, const maths::quat& q) {
            WriteRaw(out, q.x); WriteRaw(out, q.y); WriteRaw(out, q.z); WriteRaw(out, q.w);
        }

        bool ReadQuat(std::ifstream& in, maths::quat& q) {
            return ReadRaw(in, q.x) && ReadRaw(in, q.y) && ReadRaw(in, q.z) && ReadRaw(in, q.w);
        }

        void WriteAABB(std::ofstream& out, const AABB& box) {
            WriteVec3(out, box.boundsMin);
            WriteVec3(out, box.boundsMax);
        }

        bool ReadAABB(std::ifstream& in, AABB& box) {
            return ReadVec3(in, box.boundsMin) && ReadVec3(in, box.boundsMax);
        }

        // Property payload tag written ahead of each PropertyEntry's value, one-to-one with
        // PropertyValue's std::variant alternative order -- kept as an explicit enum (rather than
        // just persisting value.index() directly) so PropertyValue's alternatives can be
        // reordered in code without silently corrupting already-written .actor files.
        enum class PropertyTag : uint8_t { Bool = 0, Int32 = 1, Float = 2, Vec3 = 3, String = 4 };

        void WritePropertyEntry(std::ofstream& out, const PropertyEntry& entry) {
            WriteString(out, entry.key);
            std::visit([&out](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, bool>) {
                    WriteRaw(out, static_cast<uint8_t>(PropertyTag::Bool));
                    WriteRaw(out, value);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    WriteRaw(out, static_cast<uint8_t>(PropertyTag::Int32));
                    WriteRaw(out, value);
                } else if constexpr (std::is_same_v<T, float>) {
                    WriteRaw(out, static_cast<uint8_t>(PropertyTag::Float));
                    WriteRaw(out, value);
                } else if constexpr (std::is_same_v<T, maths::vec3>) {
                    WriteRaw(out, static_cast<uint8_t>(PropertyTag::Vec3));
                    WriteVec3(out, value);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    WriteRaw(out, static_cast<uint8_t>(PropertyTag::String));
                    WriteString(out, value);
                }
                }, entry.value);
        }

        bool ReadPropertyEntry(std::ifstream& in, PropertyEntry& entry) {
            if (!ReadString(in, entry.key)) return false;

            uint8_t tag = 0;
            if (!ReadRaw(in, tag)) return false;

            switch (static_cast<PropertyTag>(tag)) {
            case PropertyTag::Bool: {
                bool v = false;
                if (!ReadRaw(in, v)) return false;
                entry.value = v;
                return true;
            }
            case PropertyTag::Int32: {
                int32_t v = 0;
                if (!ReadRaw(in, v)) return false;
                entry.value = v;
                return true;
            }
            case PropertyTag::Float: {
                float v = 0.0f;
                if (!ReadRaw(in, v)) return false;
                entry.value = v;
                return true;
            }
            case PropertyTag::Vec3: {
                maths::vec3 v{};
                if (!ReadVec3(in, v)) return false;
                entry.value = v;
                return true;
            }
            case PropertyTag::String: {
                std::string v;
                if (!ReadString(in, v)) return false;
                entry.value = std::move(v);
                return true;
            }
            default:
                return false; // Unknown tag: newer file format than this reader understands, or corruption -- either way, fatal for this record.
            }
        }

    } // namespace

    void ActorRecord::RecomputeWorldBounds() {
        const std::array<maths::vec3, 8> corners = { {
            { localBounds.boundsMin.x, localBounds.boundsMin.y, localBounds.boundsMin.z },
            { localBounds.boundsMax.x, localBounds.boundsMin.y, localBounds.boundsMin.z },
            { localBounds.boundsMin.x, localBounds.boundsMax.y, localBounds.boundsMin.z },
            { localBounds.boundsMax.x, localBounds.boundsMax.y, localBounds.boundsMin.z },
            { localBounds.boundsMin.x, localBounds.boundsMin.y, localBounds.boundsMax.z },
            { localBounds.boundsMax.x, localBounds.boundsMin.y, localBounds.boundsMax.z },
            { localBounds.boundsMin.x, localBounds.boundsMax.y, localBounds.boundsMax.z },
            { localBounds.boundsMax.x, localBounds.boundsMax.y, localBounds.boundsMax.z },
        } };

        maths::ResetAABB(worldBounds.boundsMin, worldBounds.boundsMax);
        for (const maths::vec3& corner : corners) {
            maths::vec3 scaled{ corner.x * transform.scale.x, corner.y * transform.scale.y, corner.z * transform.scale.z };
            maths::vec3 world = RotateVectorByQuat(transform.rotation, scaled) + transform.position;
            maths::ExpandAABB(worldBounds.boundsMin, worldBounds.boundsMax, world);
        }
    }

    bool WriteActorFile(const std::filesystem::path& filePath, const ActorRecord& record) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        ActorFileHeader header;
        WriteRaw(out, header);

        WriteUuid(out, record.uuid);
        WriteUuid(out, record.parentUuid);
        WriteString(out, record.className);
        WriteString(out, record.actorLabel);

        WriteVec3(out, record.transform.position);
        WriteQuat(out, record.transform.rotation);
        WriteVec3(out, record.transform.scale);

        WriteAABB(out, record.localBounds);
        WriteAABB(out, record.worldBounds);

        WriteRaw(out, static_cast<uint32_t>(record.streamingFlags));

        WriteRaw(out, static_cast<uint32_t>(record.tags.size()));
        for (const std::string& tag : record.tags) {
            WriteString(out, tag);
        }

        WriteRaw(out, static_cast<uint32_t>(record.properties.size()));
        for (const PropertyEntry& entry : record.properties) {
            WritePropertyEntry(out, entry);
        }

        return out.good();
    }

    bool ReadActorFile(const std::filesystem::path& filePath, ActorRecord& outRecord) {
        std::ifstream in(filePath, std::ios::binary);
        if (!in.is_open()) return false;

        ActorFileHeader header;
        if (!ReadRaw(in, header)) return false;
        if (header.magic != kActorFileMagic || header.version != kActorFileVersion) return false;

        if (!ReadUuid(in, outRecord.uuid)) return false;
        if (!ReadUuid(in, outRecord.parentUuid)) return false;
        if (!ReadString(in, outRecord.className)) return false;
        if (!ReadString(in, outRecord.actorLabel)) return false;

        if (!ReadVec3(in, outRecord.transform.position)) return false;
        if (!ReadQuat(in, outRecord.transform.rotation)) return false;
        if (!ReadVec3(in, outRecord.transform.scale)) return false;

        if (!ReadAABB(in, outRecord.localBounds)) return false;
        if (!ReadAABB(in, outRecord.worldBounds)) return false;

        uint32_t streamingFlags = 0;
        if (!ReadRaw(in, streamingFlags)) return false;
        outRecord.streamingFlags = static_cast<ActorStreamingFlags>(streamingFlags);

        uint32_t tagCount = 0;
        if (!ReadRaw(in, tagCount)) return false;
        outRecord.tags.clear();
        outRecord.tags.reserve(tagCount);
        for (uint32_t i = 0; i < tagCount; ++i) {
            std::string tag;
            if (!ReadString(in, tag)) return false;
            outRecord.tags.push_back(std::move(tag));
        }

        uint32_t propertyCount = 0;
        if (!ReadRaw(in, propertyCount)) return false;
        outRecord.properties.clear();
        outRecord.properties.reserve(propertyCount);
        for (uint32_t i = 0; i < propertyCount; ++i) {
            PropertyEntry entry;
            if (!ReadPropertyEntry(in, entry)) return false;
            outRecord.properties.push_back(std::move(entry));
        }

        return true;
    }

    std::filesystem::path MakeActorFilePath(const std::filesystem::path& actorsRootDir, const Uuid& uuid) {
        std::string hex = uuid.ToHexString();
        std::string shard = hex.substr(0, 2);
        return actorsRootDir / shard / (hex + ".actor");
    }

}
