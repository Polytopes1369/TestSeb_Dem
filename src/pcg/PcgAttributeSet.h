#pragma once

// PCG framework roadmap, Phase 1 ("PCG Data Model"): a generic typed key/value data bag for
// passing arbitrary, non-spatial parameters between future PCG graph nodes (samplers/filters/
// spawners, later phases) -- UE5.8's PCG framework calls the equivalent concept an "attribute set"
// / "params data" (as opposed to point data, which is always spatial). This is deliberately NOT a
// spatial type: no position/bounds/density here, see PcgPointData.h for that.
//
// Mirrors tools/WorldPartition/OfpaActor.h's PropertyEntry/std::variant pattern (that file's own
// header comment explains the rationale, reproduced here for locality): a CLOSED std::variant, not
// an open type-erased blob, so every exhaustive std::visit over an AttributeValue is a compile
// error the moment a 6th type is added anywhere it isn't handled -- a silent runtime gap is not
// possible. maths::vec3 is included (not vec4) to match OfpaActor.h's own exact type list, keeping
// this codebase's two "generic typed property bag" implementations in lockstep; a future phase can
// extend both together if a wider type (vec4, quat) turns out to be needed.

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "core/maths/Maths.h"

namespace pcg {

    // Kept as a free-standing alias (not nested in PcgAttributeSet) so future code can name the
    // variant type directly, e.g. to write its own std::visit over a single AttributeValue without
    // needing a PcgAttributeSet instance at hand.
    using AttributeValue = std::variant<bool, int32_t, float, maths::vec3, std::string>;

    struct PcgAttributeEntry {
        std::string key;
        AttributeValue value;
    };

    // Ordered (insertion-order-preserving) key/value bag. A std::vector<PcgAttributeEntry> rather
    // than a std::map/std::unordered_map: attribute sets in practice hold a handful of entries (a
    // handful of graph-node parameters), so linear lookup is not a performance concern, and
    // preserving insertion order makes future debug/inspector UI (Phase 7's node-editor scaffold,
    // out of scope here) list attributes in the order a graph author added them rather than an
    // arbitrary hash order -- the same ergonomic reasoning ActorRecord::properties already applies
    // via OfpaActor.h's own std::vector<PropertyEntry>.
    class PcgAttributeSet {
    public:
        PcgAttributeSet() = default;

        // Upserts `key`: overwrites the existing entry's value in place (preserving its original
        // position in Entries()) if `key` is already present, otherwise appends a new entry.
        void Set(const std::string& key, AttributeValue value) {
            for (auto& entry : m_Entries) {
                if (entry.key == key) {
                    entry.value = std::move(value);
                    return;
                }
            }
            m_Entries.push_back(PcgAttributeEntry{ key, std::move(value) });
        }

        bool Has(const std::string& key) const {
            return FindEntry(key) != nullptr;
        }

        // Returns the currently-held std::variant's active index (std::variant::index()) for
        // `key`, or std::nullopt if `key` is not present -- lets a caller branch on a value's type
        // without needing to already know it (e.g. a future debug inspector rendering a generic
        // attribute list).
        std::optional<size_t> TypeIndexOf(const std::string& key) const {
            const PcgAttributeEntry* entry = FindEntry(key);
            if (!entry) {
                return std::nullopt;
            }
            return entry->value.index();
        }

        // Type-checked, non-throwing accessor: returns a pointer to the stored T if `key` exists
        // AND its currently-held variant alternative is exactly T, otherwise nullptr (mirrors
        // std::get_if's own contract, just resolved through this bag's key lookup first). No
        // exception path -- callers that want "value or crash" should dereference the pointer
        // themselves and accept that risk explicitly, matching this codebase's general aversion to
        // using exceptions for ordinary (non-corruption) control flow.
        template <typename T>
        const T* TryGet(const std::string& key) const {
            const PcgAttributeEntry* entry = FindEntry(key);
            if (!entry) {
                return nullptr;
            }
            return std::get_if<T>(&entry->value);
        }

        template <typename T>
        T* TryGet(const std::string& key) {
            PcgAttributeEntry* entry = FindEntry(key);
            if (!entry) {
                return nullptr;
            }
            return std::get_if<T>(&entry->value);
        }

        // Convenience "value or fallback" accessor, for call sites that would rather not branch on
        // a null pointer -- returns `fallback` both when `key` is absent AND when it is present but
        // holds a different type than T (a caller that needs to distinguish those two cases should
        // use TryGet/Has/TypeIndexOf instead).
        template <typename T>
        T GetOr(const std::string& key, T fallback) const {
            const T* found = TryGet<T>(key);
            return found ? *found : fallback;
        }

        // Erases the entry for `key` if present. Returns true if an entry was actually removed.
        bool Remove(const std::string& key) {
            for (auto it = m_Entries.begin(); it != m_Entries.end(); ++it) {
                if (it->key == key) {
                    m_Entries.erase(it);
                    return true;
                }
            }
            return false;
        }

        void Clear() {
            m_Entries.clear();
        }

        size_t Size() const {
            return m_Entries.size();
        }

        // Full insertion-ordered entry list -- exposed read-only for iteration (e.g. a future
        // debug inspector or an offline serializer wanting to walk every attribute), same
        // "expose the backing vector directly" convention ActorRecord::properties already uses.
        const std::vector<PcgAttributeEntry>& Entries() const {
            return m_Entries;
        }

    private:
        const PcgAttributeEntry* FindEntry(const std::string& key) const {
            for (const auto& entry : m_Entries) {
                if (entry.key == key) {
                    return &entry;
                }
            }
            return nullptr;
        }

        PcgAttributeEntry* FindEntry(const std::string& key) {
            for (auto& entry : m_Entries) {
                if (entry.key == key) {
                    return &entry;
                }
            }
            return nullptr;
        }

        std::vector<PcgAttributeEntry> m_Entries;
    };

}
