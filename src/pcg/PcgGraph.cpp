// PCG framework roadmap, Phase 5.1 ("PCG Graph Engine Core" -- graph data model + serialization).
// See PcgGraph.h for the full design rationale (layering vs. PcgGraphEvaluator.h, closed-variant
// PcgPinData, why node types are strings). This file implements graph mutation (AddNode/AddLink/
// RemoveNode/RemoveLink), cycle/type-compatibility validation, and JSON serialization.
//
// JSON library choice: vendor/imgui-node-editor/crude_json.h/.cpp, already vendored into this
// codebase for Phase 7.1's PcgGraphEditorPanel (thedmd/imgui-node-editor's own dependency) --
// reused here rather than writing a second hand-rolled JSON library, per this phase's own design
// brief. crude_json::value is a small variant-like type (null/object/array/string/boolean/number,
// object backed by std::map<string,value>) with a recursive-descent parser (value::parse) and a
// pretty-printing dumper (value::dump) -- exactly what a "simple, human-inspectable format" needs,
// no more. Its const operator[](key) asserts (and std::terminate()s in a build where assert is
// compiled out) if the key is missing, so every read in this file goes through the small
// contains()-then-index FindField()/FindArrayField() helpers below instead of raw operator[], to
// stay defensive against hand-edited or corrupted JSON text during DeserializeFromJson().
//
// JSON schema (top-level = one PcgGraph):
//   {
//     "nextNodeId": <number>,
//     "nodes": [ <node>, ... ],
//     "links": [ <link>, ... ]
//   }
//   <node> = {
//     "id": <number>, "typeId": <string>, "displayName": <string>,
//     "inputPins": [ <pin>, ... ], "outputPins": [ <pin>, ... ],
//     "params": [ <attributeEntry>, ... ],
//     "subgraph": null | <graph>,                      -- non-null only for kSubgraphNodeTypeId nodes
//     "subgraphInputBindings": [ <binding>, ... ], "subgraphOutputBindings": [ <binding>, ... ]
//   }
//   <pin> = { "name": <string>, "type": <string, one of ToString(PcgPinDataType)>, "required": <bool> }
//   <attributeEntry> = { "key": <string>, "type": "bool"|"int32"|"float"|"vec3"|"string", "value": ... }
//     (vec3's "value" is itself { "x":, "y":, "z": }; every other type's "value" is a plain JSON
//     scalar of the obvious kind)
//   <link> = { "sourceNodeId":, "sourcePinName":, "destNodeId":, "destPinName": }
//   <binding> = { "outerPinName":, "innerNodeId":, "innerPinName": }
// Every list above is a JSON ARRAY (not an object keyed by name/id), specifically so element order
// is preserved exactly through a round-trip -- see PcgGraph::StructurallyEqual()'s own comment in
// the header for why that lets structural equality be a plain order-sensitive comparison.

#include "pcg/PcgGraph.h"

#include "../../vendor/imgui-node-editor/crude_json.h"

#include <algorithm>
#include <cassert>
#include <deque>
#include <unordered_set>

namespace pcg {

    namespace {

        // --- crude_json defensive-read helpers -------------------------------------------------
        // crude_json::value's const operator[](key) CRUDE_ASSERT()s the key exists and then
        // std::terminate()s if that assert is compiled out (NDEBUG) -- not acceptable for parsing
        // externally-editable JSON text. These helpers return nullptr/false instead.

        const crude_json::value* FindField(const crude_json::value& obj, const std::string& key) {
            if (!obj.is_object() || !obj.contains(key)) {
                return nullptr;
            }
            return &obj[key];
        }

        bool ReadString(const crude_json::value& obj, const std::string& key, std::string& out) {
            const crude_json::value* field = FindField(obj, key);
            const crude_json::string* str = field ? field->get_ptr<crude_json::string>() : nullptr;
            if (!str) {
                return false;
            }
            out = *str;
            return true;
        }

        bool ReadNumber(const crude_json::value& obj, const std::string& key, double& out) {
            const crude_json::value* field = FindField(obj, key);
            const crude_json::number* num = field ? field->get_ptr<crude_json::number>() : nullptr;
            if (!num) {
                return false;
            }
            out = *num;
            return true;
        }

        bool ReadUint32(const crude_json::value& obj, const std::string& key, uint32_t& out) {
            double num = 0.0;
            if (!ReadNumber(obj, key, num)) {
                return false;
            }
            out = static_cast<uint32_t>(num);
            return true;
        }

        bool ReadBool(const crude_json::value& obj, const std::string& key, bool& out) {
            const crude_json::value* field = FindField(obj, key);
            const crude_json::boolean* b = field ? field->get_ptr<crude_json::boolean>() : nullptr;
            if (!b) {
                return false;
            }
            out = *b;
            return true;
        }

        // Returns nullptr if `key` is absent/wrong-type; a valid array pointer otherwise. Absence
        // of an expected array is treated as "empty list", not a hard parse error, by every call
        // site below (keeps the schema forward-tolerant: an older saved graph with no
        // "subgraphInputBindings" field at all still loads fine as a node with zero bindings).
        const crude_json::array* ReadArrayField(const crude_json::value& obj, const std::string& key) {
            const crude_json::value* field = FindField(obj, key);
            return field ? field->get_ptr<crude_json::array>() : nullptr;
        }

        // --- PcgPinDataType <-> string -----------------------------------------------------------

        struct PinTypeNameEntry { PcgPinDataType type; const char* name; };
        constexpr PinTypeNameEntry kPinTypeNames[] = {
            { PcgPinDataType::Points,        "Points" },
            { PcgPinDataType::AttributeSet,  "AttributeSet" },
            { PcgPinDataType::Surface,       "Surface" },
            { PcgPinDataType::Volume,        "Volume" },
            { PcgPinDataType::Landscape,     "Landscape" },
            { PcgPinDataType::Spline,        "Spline" },
            { PcgPinDataType::SpawnRequests, "SpawnRequests" }, // Phase 4.1 additive extension.
        };

        // --- AttributeValue <-> json --------------------------------------------------------------

        crude_json::value AttributeValueToJson(const AttributeValue& value) {
            crude_json::object obj;
            switch (value.index()) {
            case 0: // bool
                obj["type"] = crude_json::value(std::string("bool"));
                obj["value"] = crude_json::value(std::get<bool>(value));
                break;
            case 1: // int32_t
                obj["type"] = crude_json::value(std::string("int32"));
                obj["value"] = crude_json::value(static_cast<crude_json::number>(std::get<int32_t>(value)));
                break;
            case 2: // float
                obj["type"] = crude_json::value(std::string("float"));
                obj["value"] = crude_json::value(static_cast<crude_json::number>(std::get<float>(value)));
                break;
            case 3: { // maths::vec3
                const maths::vec3& v = std::get<maths::vec3>(value);
                crude_json::object vecObj;
                vecObj["x"] = crude_json::value(static_cast<crude_json::number>(v.x));
                vecObj["y"] = crude_json::value(static_cast<crude_json::number>(v.y));
                vecObj["z"] = crude_json::value(static_cast<crude_json::number>(v.z));
                obj["type"] = crude_json::value(std::string("vec3"));
                obj["value"] = crude_json::value(std::move(vecObj));
                break;
            }
            case 4: // std::string
                obj["type"] = crude_json::value(std::string("string"));
                obj["value"] = crude_json::value(std::get<std::string>(value));
                break;
            default:
                assert(false && "AttributeValueToJson: unhandled AttributeValue alternative");
                break;
            }
            return crude_json::value(std::move(obj));
        }

        bool AttributeValueFromJson(const crude_json::value& json, AttributeValue& outValue, std::string* outError) {
            std::string typeName;
            const crude_json::value* valueField = FindField(json, "value");
            if (!ReadString(json, "type", typeName) || !valueField) {
                if (outError) *outError = "attribute entry missing 'type' or 'value' field";
                return false;
            }
            if (typeName == "bool") {
                const crude_json::boolean* b = valueField->get_ptr<crude_json::boolean>();
                if (!b) { if (outError) *outError = "attribute 'value' is not a bool"; return false; }
                outValue = *b;
                return true;
            }
            if (typeName == "int32") {
                const crude_json::number* n = valueField->get_ptr<crude_json::number>();
                if (!n) { if (outError) *outError = "attribute 'value' is not a number (int32)"; return false; }
                outValue = static_cast<int32_t>(*n);
                return true;
            }
            if (typeName == "float") {
                const crude_json::number* n = valueField->get_ptr<crude_json::number>();
                if (!n) { if (outError) *outError = "attribute 'value' is not a number (float)"; return false; }
                outValue = static_cast<float>(*n);
                return true;
            }
            if (typeName == "vec3") {
                double x = 0.0, y = 0.0, z = 0.0;
                if (!ReadNumber(*valueField, "x", x) || !ReadNumber(*valueField, "y", y) || !ReadNumber(*valueField, "z", z)) {
                    if (outError) *outError = "attribute 'value' (vec3) missing x/y/z";
                    return false;
                }
                outValue = maths::vec3{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };
                return true;
            }
            if (typeName == "string") {
                const crude_json::string* s = valueField->get_ptr<crude_json::string>();
                if (!s) { if (outError) *outError = "attribute 'value' is not a string"; return false; }
                outValue = *s;
                return true;
            }
            if (outError) *outError = "attribute entry has unknown 'type': " + typeName;
            return false;
        }

        // --- Structural equality helpers (order-sensitive, see StructurallyEqual's own comment) --

        bool NearlyEqualFloat(float a, float b, float epsilon = 1.0e-5f) {
            return std::abs(a - b) <= epsilon;
        }

        bool AttributeValuesEqual(const AttributeValue& a, const AttributeValue& b) {
            if (a.index() != b.index()) {
                return false;
            }
            switch (a.index()) {
            case 0: return std::get<bool>(a) == std::get<bool>(b);
            case 1: return std::get<int32_t>(a) == std::get<int32_t>(b);
            case 2: return NearlyEqualFloat(std::get<float>(a), std::get<float>(b));
            case 3: {
                const maths::vec3& va = std::get<maths::vec3>(a);
                const maths::vec3& vb = std::get<maths::vec3>(b);
                return NearlyEqualFloat(va.x, vb.x) && NearlyEqualFloat(va.y, vb.y) && NearlyEqualFloat(va.z, vb.z);
            }
            case 4: return std::get<std::string>(a) == std::get<std::string>(b);
            default: return false;
            }
        }

        bool AttributeSetsEqual(const PcgAttributeSet& a, const PcgAttributeSet& b) {
            if (a.Entries().size() != b.Entries().size()) {
                return false;
            }
            for (size_t i = 0; i < a.Entries().size(); ++i) {
                const PcgAttributeEntry& ea = a.Entries()[i];
                const PcgAttributeEntry& eb = b.Entries()[i];
                if (ea.key != eb.key || !AttributeValuesEqual(ea.value, eb.value)) {
                    return false;
                }
            }
            return true;
        }

        bool PinDescsEqual(const PcgPinDesc& a, const PcgPinDesc& b) {
            return a.name == b.name && a.type == b.type && a.required == b.required;
        }

        bool PinDescListsEqual(const std::vector<PcgPinDesc>& a, const std::vector<PcgPinDesc>& b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!PinDescsEqual(a[i], b[i])) return false;
            }
            return true;
        }

        bool SubgraphBindingsEqual(const std::vector<PcgNode::SubgraphPinBinding>& a, const std::vector<PcgNode::SubgraphPinBinding>& b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].outerPinName != b[i].outerPinName || a[i].innerNodeId != b[i].innerNodeId || a[i].innerPinName != b[i].innerPinName) {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    // ============================================================================================
    // PcgPinDataType <-> string / compatibility
    // ============================================================================================

    const char* ToString(PcgPinDataType type) {
        for (const PinTypeNameEntry& entry : kPinTypeNames) {
            if (entry.type == type) {
                return entry.name;
            }
        }
        assert(false && "ToString(PcgPinDataType): unhandled enumerator");
        return "Unknown";
    }

    std::optional<PcgPinDataType> PinDataTypeFromString(const std::string& text) {
        for (const PinTypeNameEntry& entry : kPinTypeNames) {
            if (text == entry.name) {
                return entry.type;
            }
        }
        return std::nullopt;
    }

    PcgPinDataType PinDataTypeOfValue(const PcgPinData& data) {
        switch (data.index()) {
        case 1: return PcgPinDataType::Points;
        case 2: return PcgPinDataType::AttributeSet;
        case 3: return PcgPinDataType::Surface;
        case 4: return PcgPinDataType::Volume;
        case 5: return PcgPinDataType::Landscape;
        case 6: return PcgPinDataType::Spline;
        case 7: return PcgPinDataType::SpawnRequests; // Phase 4.1 additive extension.
        default:
            assert(false && "PinDataTypeOfValue: called on an empty (monostate) PcgPinData");
            return PcgPinDataType::Points;
        }
    }

    bool ArePinTypesCompatible(PcgPinDataType output, PcgPinDataType input) {
        // No implicit conversions exist yet -- see this function's own header comment for why it
        // is still a dedicated function rather than an inline `==` at every call site.
        return output == input;
    }

    // ============================================================================================
    // PcgGraph: node/link mutation
    // ============================================================================================

    uint32_t PcgGraph::AddNode(PcgNodeTypeId typeId, std::vector<PcgPinDesc> inputPins, std::vector<PcgPinDesc> outputPins,
        PcgAttributeSet params, std::string displayName) {
        PcgNode node;
        node.id = m_NextNodeId++;
        node.typeId = std::move(typeId);
        node.displayName = std::move(displayName);
        node.inputPins = std::move(inputPins);
        node.outputPins = std::move(outputPins);
        node.params = std::move(params);
        m_Nodes.push_back(std::move(node));
        return m_Nodes.back().id;
    }

    uint32_t PcgGraph::AddNodeWithId(PcgNode node) {
        const uint32_t id = node.id;
        m_Nodes.push_back(std::move(node));
        if (id >= m_NextNodeId) {
            m_NextNodeId = id + 1;
        }
        return id;
    }

    bool PcgGraph::RemoveNode(uint32_t nodeId) {
        const auto nodeIt = std::find_if(m_Nodes.begin(), m_Nodes.end(), [nodeId](const PcgNode& n) { return n.id == nodeId; });
        if (nodeIt == m_Nodes.end()) {
            return false;
        }
        m_Nodes.erase(nodeIt);

        // Cascade: drop every link that touched this node as either endpoint, so the graph never
        // ends up holding a dangling link referencing a node id that no longer exists.
        m_Links.erase(
            std::remove_if(m_Links.begin(), m_Links.end(),
                [nodeId](const PcgLink& link) { return link.sourceNodeId == nodeId || link.destNodeId == nodeId; }),
            m_Links.end());
        return true;
    }

    const PcgNode* PcgGraph::FindNode(uint32_t nodeId) const {
        const auto it = std::find_if(m_Nodes.begin(), m_Nodes.end(), [nodeId](const PcgNode& n) { return n.id == nodeId; });
        return it != m_Nodes.end() ? &(*it) : nullptr;
    }

    PcgNode* PcgGraph::FindNode(uint32_t nodeId) {
        const auto it = std::find_if(m_Nodes.begin(), m_Nodes.end(), [nodeId](PcgNode& n) { return n.id == nodeId; });
        return it != m_Nodes.end() ? &(*it) : nullptr;
    }

    bool PcgGraph::HasPath(uint32_t fromNodeId, uint32_t toNodeId) const {
        if (fromNodeId == toNodeId) {
            return true;
        }
        std::unordered_set<uint32_t> visited;
        std::deque<uint32_t> queue;
        queue.push_back(fromNodeId);
        visited.insert(fromNodeId);
        while (!queue.empty()) {
            const uint32_t current = queue.front();
            queue.pop_front();
            for (const PcgLink& link : m_Links) {
                if (link.sourceNodeId != current) {
                    continue;
                }
                if (link.destNodeId == toNodeId) {
                    return true;
                }
                if (visited.insert(link.destNodeId).second) {
                    queue.push_back(link.destNodeId);
                }
            }
        }
        return false;
    }

    std::string PcgGraph::DescribeAddLinkStatus(AddLinkStatus status) {
        switch (status) {
        case AddLinkStatus::Ok: return "link added successfully";
        case AddLinkStatus::UnknownSourceNode: return "source node id does not exist";
        case AddLinkStatus::UnknownDestNode: return "destination node id does not exist";
        case AddLinkStatus::UnknownSourcePin: return "source node has no output pin with that name";
        case AddLinkStatus::UnknownDestPin: return "destination node has no input pin with that name";
        case AddLinkStatus::TypeMismatch: return "source output pin type is not compatible with destination input pin type";
        case AddLinkStatus::DestPinAlreadyConnected: return "destination input pin already has an incoming link (an input pin accepts at most one link)";
        case AddLinkStatus::WouldCreateCycle: return "this link would create a cycle; a PCG graph must be a DAG";
        }
        return "unknown AddLinkStatus";
    }

    PcgGraph::AddLinkStatus PcgGraph::AddLink(uint32_t sourceNodeId, const std::string& sourcePinName,
        uint32_t destNodeId, const std::string& destPinName, std::string* outMessage) {
        const PcgNode* sourceNode = FindNode(sourceNodeId);
        if (!sourceNode) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::UnknownSourceNode);
            return AddLinkStatus::UnknownSourceNode;
        }
        const PcgNode* destNode = FindNode(destNodeId);
        if (!destNode) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::UnknownDestNode);
            return AddLinkStatus::UnknownDestNode;
        }

        const auto sourcePinIt = std::find_if(sourceNode->outputPins.begin(), sourceNode->outputPins.end(),
            [&sourcePinName](const PcgPinDesc& p) { return p.name == sourcePinName; });
        if (sourcePinIt == sourceNode->outputPins.end()) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::UnknownSourcePin) + " ('" + sourcePinName + "')";
            return AddLinkStatus::UnknownSourcePin;
        }
        const auto destPinIt = std::find_if(destNode->inputPins.begin(), destNode->inputPins.end(),
            [&destPinName](const PcgPinDesc& p) { return p.name == destPinName; });
        if (destPinIt == destNode->inputPins.end()) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::UnknownDestPin) + " ('" + destPinName + "')";
            return AddLinkStatus::UnknownDestPin;
        }

        if (!ArePinTypesCompatible(sourcePinIt->type, destPinIt->type)) {
            if (outMessage) {
                *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::TypeMismatch) +
                    " (" + ToString(sourcePinIt->type) + " -> " + ToString(destPinIt->type) + ")";
            }
            return AddLinkStatus::TypeMismatch;
        }

        const bool destAlreadyConnected = std::any_of(m_Links.begin(), m_Links.end(),
            [destNodeId, &destPinName](const PcgLink& link) { return link.destNodeId == destNodeId && link.destPinName == destPinName; });
        if (destAlreadyConnected) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::DestPinAlreadyConnected);
            return AddLinkStatus::DestPinAlreadyConnected;
        }

        // Cycle check: adding sourceNodeId -> destNodeId would close a cycle iff destNodeId can
        // already reach sourceNodeId via existing links (see HasPath's own comment).
        if (HasPath(destNodeId, sourceNodeId)) {
            if (outMessage) *outMessage = "AddLink failed: " + DescribeAddLinkStatus(AddLinkStatus::WouldCreateCycle);
            return AddLinkStatus::WouldCreateCycle;
        }

        m_Links.push_back(PcgLink{ sourceNodeId, sourcePinName, destNodeId, destPinName });
        if (outMessage) *outMessage = DescribeAddLinkStatus(AddLinkStatus::Ok);
        return AddLinkStatus::Ok;
    }

    bool PcgGraph::RemoveLink(uint32_t sourceNodeId, const std::string& sourcePinName, uint32_t destNodeId, const std::string& destPinName) {
        const auto it = std::find_if(m_Links.begin(), m_Links.end(), [&](const PcgLink& link) {
            return link.sourceNodeId == sourceNodeId && link.sourcePinName == sourcePinName &&
                link.destNodeId == destNodeId && link.destPinName == destPinName;
            });
        if (it == m_Links.end()) {
            return false;
        }
        m_Links.erase(it);
        return true;
    }

    // ============================================================================================
    // Serialization
    // ============================================================================================

    crude_json::value PcgGraph::PinDescToJsonValue(const PcgPinDesc& pin) {
        crude_json::object obj;
        obj["name"] = crude_json::value(pin.name);
        obj["type"] = crude_json::value(std::string(ToString(pin.type)));
        obj["required"] = crude_json::value(pin.required);
        return crude_json::value(std::move(obj));
    }

    bool PcgGraph::PinDescFromJsonValue(const crude_json::value& json, PcgPinDesc& outPin, std::string* outError) {
        std::string typeName;
        if (!ReadString(json, "name", outPin.name)) {
            if (outError) *outError = "pin descriptor missing 'name'";
            return false;
        }
        if (!ReadString(json, "type", typeName)) {
            if (outError) *outError = "pin descriptor '" + outPin.name + "' missing 'type'";
            return false;
        }
        const std::optional<PcgPinDataType> type = PinDataTypeFromString(typeName);
        if (!type) {
            if (outError) *outError = "pin descriptor '" + outPin.name + "' has unknown type '" + typeName + "'";
            return false;
        }
        outPin.type = *type;
        if (!ReadBool(json, "required", outPin.required)) {
            outPin.required = true; // Forward-tolerant default for an older save missing this field.
        }
        return true;
    }

    crude_json::value PcgGraph::AttributeSetToJsonValue(const PcgAttributeSet& attrs) {
        crude_json::array arr;
        for (const PcgAttributeEntry& entry : attrs.Entries()) {
            crude_json::object obj;
            obj["key"] = crude_json::value(entry.key);
            crude_json::value valueJson = AttributeValueToJson(entry.value);
            // AttributeValueToJson() already produced {"type":..,"value":..} -- merge it directly
            // into this entry's object rather than nesting it under a third key, so a hand-written
            // save file's attribute entries read as flat {"key","type","value"} triples.
            obj["type"] = valueJson["type"];
            obj["value"] = valueJson["value"];
            arr.push_back(crude_json::value(std::move(obj)));
        }
        return crude_json::value(std::move(arr));
    }

    bool PcgGraph::AttributeSetFromJsonValue(const crude_json::value& json, PcgAttributeSet& outAttrs, std::string* outError) {
        outAttrs.Clear();
        const crude_json::array* arr = json.get_ptr<crude_json::array>();
        if (!arr) {
            if (outError) *outError = "params field is not a JSON array";
            return false;
        }
        for (const crude_json::value& entryJson : *arr) {
            std::string key;
            if (!ReadString(entryJson, "key", key)) {
                if (outError) *outError = "attribute entry missing 'key'";
                return false;
            }
            AttributeValue value;
            std::string entryError;
            if (!AttributeValueFromJson(entryJson, value, &entryError)) {
                if (outError) *outError = "attribute '" + key + "': " + entryError;
                return false;
            }
            outAttrs.Set(key, std::move(value));
        }
        return true;
    }

    crude_json::value PcgGraph::LinkToJsonValue(const PcgLink& link) {
        crude_json::object obj;
        obj["sourceNodeId"] = crude_json::value(static_cast<crude_json::number>(link.sourceNodeId));
        obj["sourcePinName"] = crude_json::value(link.sourcePinName);
        obj["destNodeId"] = crude_json::value(static_cast<crude_json::number>(link.destNodeId));
        obj["destPinName"] = crude_json::value(link.destPinName);
        return crude_json::value(std::move(obj));
    }

    bool PcgGraph::LinkFromJsonValue(const crude_json::value& json, PcgLink& outLink, std::string* outError) {
        if (!ReadUint32(json, "sourceNodeId", outLink.sourceNodeId) ||
            !ReadString(json, "sourcePinName", outLink.sourcePinName) ||
            !ReadUint32(json, "destNodeId", outLink.destNodeId) ||
            !ReadString(json, "destPinName", outLink.destPinName)) {
            if (outError) *outError = "link entry missing one of sourceNodeId/sourcePinName/destNodeId/destPinName";
            return false;
        }
        return true;
    }

    crude_json::value PcgGraph::NodeToJsonValue(const PcgNode& node) {
        crude_json::object obj;
        obj["id"] = crude_json::value(static_cast<crude_json::number>(node.id));
        obj["typeId"] = crude_json::value(node.typeId);
        obj["displayName"] = crude_json::value(node.displayName);

        crude_json::array inputPinsArr;
        for (const PcgPinDesc& pin : node.inputPins) inputPinsArr.push_back(PinDescToJsonValue(pin));
        obj["inputPins"] = crude_json::value(std::move(inputPinsArr));

        crude_json::array outputPinsArr;
        for (const PcgPinDesc& pin : node.outputPins) outputPinsArr.push_back(PinDescToJsonValue(pin));
        obj["outputPins"] = crude_json::value(std::move(outputPinsArr));

        obj["params"] = AttributeSetToJsonValue(node.params);

        obj["subgraph"] = node.subgraph ? node.subgraph->ToJsonValue() : crude_json::value(crude_json::type_t::null);

        crude_json::array inputBindingsArr;
        for (const PcgNode::SubgraphPinBinding& binding : node.subgraphInputBindings) {
            crude_json::object bindingObj;
            bindingObj["outerPinName"] = crude_json::value(binding.outerPinName);
            bindingObj["innerNodeId"] = crude_json::value(static_cast<crude_json::number>(binding.innerNodeId));
            bindingObj["innerPinName"] = crude_json::value(binding.innerPinName);
            inputBindingsArr.push_back(crude_json::value(std::move(bindingObj)));
        }
        obj["subgraphInputBindings"] = crude_json::value(std::move(inputBindingsArr));

        crude_json::array outputBindingsArr;
        for (const PcgNode::SubgraphPinBinding& binding : node.subgraphOutputBindings) {
            crude_json::object bindingObj;
            bindingObj["outerPinName"] = crude_json::value(binding.outerPinName);
            bindingObj["innerNodeId"] = crude_json::value(static_cast<crude_json::number>(binding.innerNodeId));
            bindingObj["innerPinName"] = crude_json::value(binding.innerPinName);
            outputBindingsArr.push_back(crude_json::value(std::move(bindingObj)));
        }
        obj["subgraphOutputBindings"] = crude_json::value(std::move(outputBindingsArr));

        return crude_json::value(std::move(obj));
    }

    bool PcgGraph::NodeFromJsonValue(const crude_json::value& json, PcgNode& outNode, std::string* outError) {
        if (!ReadUint32(json, "id", outNode.id)) {
            if (outError) *outError = "node missing 'id'";
            return false;
        }
        if (!ReadString(json, "typeId", outNode.typeId)) {
            if (outError) *outError = "node " + std::to_string(outNode.id) + " missing 'typeId'";
            return false;
        }
        ReadString(json, "displayName", outNode.displayName); // Optional; empty is a valid default.

        const crude_json::array* inputPinsArr = ReadArrayField(json, "inputPins");
        if (!inputPinsArr) {
            if (outError) *outError = "node " + std::to_string(outNode.id) + " missing 'inputPins' array";
            return false;
        }
        for (const crude_json::value& pinJson : *inputPinsArr) {
            PcgPinDesc pin;
            std::string pinError;
            if (!PinDescFromJsonValue(pinJson, pin, &pinError)) {
                if (outError) *outError = "node " + std::to_string(outNode.id) + " inputPins: " + pinError;
                return false;
            }
            outNode.inputPins.push_back(std::move(pin));
        }

        const crude_json::array* outputPinsArr = ReadArrayField(json, "outputPins");
        if (!outputPinsArr) {
            if (outError) *outError = "node " + std::to_string(outNode.id) + " missing 'outputPins' array";
            return false;
        }
        for (const crude_json::value& pinJson : *outputPinsArr) {
            PcgPinDesc pin;
            std::string pinError;
            if (!PinDescFromJsonValue(pinJson, pin, &pinError)) {
                if (outError) *outError = "node " + std::to_string(outNode.id) + " outputPins: " + pinError;
                return false;
            }
            outNode.outputPins.push_back(std::move(pin));
        }

        const crude_json::value* paramsField = FindField(json, "params");
        if (paramsField) {
            std::string paramsError;
            if (!AttributeSetFromJsonValue(*paramsField, outNode.params, &paramsError)) {
                if (outError) *outError = "node " + std::to_string(outNode.id) + " params: " + paramsError;
                return false;
            }
        }

        const crude_json::value* subgraphField = FindField(json, "subgraph");
        if (subgraphField && !subgraphField->is_null()) {
            std::string subgraphError;
            std::optional<PcgGraph> nested = FromJsonValue(*subgraphField, &subgraphError);
            if (!nested) {
                if (outError) *outError = "node " + std::to_string(outNode.id) + " subgraph: " + subgraphError;
                return false;
            }
            outNode.subgraph = std::make_shared<PcgGraph>(std::move(*nested));
        }

        const crude_json::array* inputBindingsArr = ReadArrayField(json, "subgraphInputBindings");
        if (inputBindingsArr) {
            for (const crude_json::value& bindingJson : *inputBindingsArr) {
                PcgNode::SubgraphPinBinding binding;
                if (!ReadString(bindingJson, "outerPinName", binding.outerPinName) ||
                    !ReadUint32(bindingJson, "innerNodeId", binding.innerNodeId) ||
                    !ReadString(bindingJson, "innerPinName", binding.innerPinName)) {
                    if (outError) *outError = "node " + std::to_string(outNode.id) + " subgraphInputBindings entry malformed";
                    return false;
                }
                outNode.subgraphInputBindings.push_back(std::move(binding));
            }
        }

        const crude_json::array* outputBindingsArr = ReadArrayField(json, "subgraphOutputBindings");
        if (outputBindingsArr) {
            for (const crude_json::value& bindingJson : *outputBindingsArr) {
                PcgNode::SubgraphPinBinding binding;
                if (!ReadString(bindingJson, "outerPinName", binding.outerPinName) ||
                    !ReadUint32(bindingJson, "innerNodeId", binding.innerNodeId) ||
                    !ReadString(bindingJson, "innerPinName", binding.innerPinName)) {
                    if (outError) *outError = "node " + std::to_string(outNode.id) + " subgraphOutputBindings entry malformed";
                    return false;
                }
                outNode.subgraphOutputBindings.push_back(std::move(binding));
            }
        }

        return true;
    }

    crude_json::value PcgGraph::ToJsonValue() const {
        crude_json::object obj;
        obj["nextNodeId"] = crude_json::value(static_cast<crude_json::number>(m_NextNodeId));

        crude_json::array nodesArr;
        for (const PcgNode& node : m_Nodes) nodesArr.push_back(NodeToJsonValue(node));
        obj["nodes"] = crude_json::value(std::move(nodesArr));

        crude_json::array linksArr;
        for (const PcgLink& link : m_Links) linksArr.push_back(LinkToJsonValue(link));
        obj["links"] = crude_json::value(std::move(linksArr));

        return crude_json::value(std::move(obj));
    }

    std::optional<PcgGraph> PcgGraph::FromJsonValue(const crude_json::value& json, std::string* outError) {
        if (!json.is_object()) {
            if (outError) *outError = "graph JSON root is not an object";
            return std::nullopt;
        }

        PcgGraph graph;

        uint32_t nextNodeId = 1;
        ReadUint32(json, "nextNodeId", nextNodeId); // Optional; recomputed from node ids below regardless.

        const crude_json::array* nodesArr = ReadArrayField(json, "nodes");
        if (!nodesArr) {
            if (outError) *outError = "graph JSON missing 'nodes' array";
            return std::nullopt;
        }
        for (const crude_json::value& nodeJson : *nodesArr) {
            PcgNode node;
            std::string nodeError;
            if (!NodeFromJsonValue(nodeJson, node, &nodeError)) {
                if (outError) *outError = nodeError;
                return std::nullopt;
            }
            graph.AddNodeWithId(std::move(node));
        }
        if (nextNodeId > graph.m_NextNodeId) {
            graph.m_NextNodeId = nextNodeId;
        }

        const crude_json::array* linksArr = ReadArrayField(json, "links");
        if (!linksArr) {
            if (outError) *outError = "graph JSON missing 'links' array";
            return std::nullopt;
        }
        for (const crude_json::value& linkJson : *linksArr) {
            PcgLink link;
            std::string linkError;
            if (!LinkFromJsonValue(linkJson, link, &linkError)) {
                if (outError) *outError = linkError;
                return std::nullopt;
            }
            // Re-validate through the normal public AddLink() -- proves the deserialized graph is
            // ACTUALLY a valid DAG with type-consistent links, not just well-formed JSON. A
            // corrupted/hand-edited save (e.g. a manually re-added cyclic link) is rejected here
            // with a clear error rather than silently producing an invalid PcgGraph.
            std::string addError;
            const PcgGraph::AddLinkStatus status = graph.AddLink(link.sourceNodeId, link.sourcePinName, link.destNodeId, link.destPinName, &addError);
            if (status != PcgGraph::AddLinkStatus::Ok) {
                if (outError) *outError = "graph JSON link failed to re-validate: " + addError;
                return std::nullopt;
            }
        }

        return graph;
    }

    std::string PcgGraph::SerializeToJson(int indent) const {
        return ToJsonValue().dump(indent);
    }

    std::optional<PcgGraph> PcgGraph::DeserializeFromJson(const std::string& jsonText, std::string* outError) {
        crude_json::value parsed = crude_json::value::parse(jsonText);
        if (parsed.is_discarded()) {
            if (outError) *outError = "invalid JSON text (parse failed)";
            return std::nullopt;
        }
        return FromJsonValue(parsed, outError);
    }

    // ============================================================================================
    // Structural equality
    // ============================================================================================

    bool PcgGraph::StructurallyEqual(const PcgGraph& a, const PcgGraph& b, std::string* outDifference) {
        if (a.m_NextNodeId != b.m_NextNodeId) {
            if (outDifference) *outDifference = "nextNodeId differs";
            return false;
        }
        if (a.m_Nodes.size() != b.m_Nodes.size()) {
            if (outDifference) *outDifference = "node count differs";
            return false;
        }
        for (size_t i = 0; i < a.m_Nodes.size(); ++i) {
            const PcgNode& na = a.m_Nodes[i];
            const PcgNode& nb = b.m_Nodes[i];
            if (na.id != nb.id || na.typeId != nb.typeId || na.displayName != nb.displayName) {
                if (outDifference) *outDifference = "node[" + std::to_string(i) + "] id/typeId/displayName differs";
                return false;
            }
            if (!PinDescListsEqual(na.inputPins, nb.inputPins) || !PinDescListsEqual(na.outputPins, nb.outputPins)) {
                if (outDifference) *outDifference = "node[" + std::to_string(i) + "] pin shape differs";
                return false;
            }
            if (!AttributeSetsEqual(na.params, nb.params)) {
                if (outDifference) *outDifference = "node[" + std::to_string(i) + "] params differ";
                return false;
            }
            if (!SubgraphBindingsEqual(na.subgraphInputBindings, nb.subgraphInputBindings) ||
                !SubgraphBindingsEqual(na.subgraphOutputBindings, nb.subgraphOutputBindings)) {
                if (outDifference) *outDifference = "node[" + std::to_string(i) + "] subgraph bindings differ";
                return false;
            }
            const bool hasSubgraphA = static_cast<bool>(na.subgraph);
            const bool hasSubgraphB = static_cast<bool>(nb.subgraph);
            if (hasSubgraphA != hasSubgraphB) {
                if (outDifference) *outDifference = "node[" + std::to_string(i) + "] subgraph presence differs";
                return false;
            }
            if (hasSubgraphA && !StructurallyEqual(*na.subgraph, *nb.subgraph, outDifference)) {
                return false;
            }
        }

        if (a.m_Links.size() != b.m_Links.size()) {
            if (outDifference) *outDifference = "link count differs";
            return false;
        }
        for (size_t i = 0; i < a.m_Links.size(); ++i) {
            const PcgLink& la = a.m_Links[i];
            const PcgLink& lb = b.m_Links[i];
            if (la.sourceNodeId != lb.sourceNodeId || la.sourcePinName != lb.sourcePinName ||
                la.destNodeId != lb.destNodeId || la.destPinName != lb.destPinName) {
                if (outDifference) *outDifference = "link[" + std::to_string(i) + "] differs";
                return false;
            }
        }

        return true;
    }

}
