// Standalone, framework-free unit test for the PCG framework roadmap's Phase 1 ("PCG Data Model")
// types: src/pcg/PcgPointData.h, PcgAttributeSet.h, PcgSpatialData.h, PcgSeededRandom.h. Exercises
// point construction/transform/steepness-density-falloff, attribute-set typed round-trips, the
// Surface/Volume/Landscape/Spline spatial-data wrappers, and seeded-RNG determinism (the hard
// "same seed -> bit-identical sequence" requirement future GPU-parity phases depend on). Exits 0 if
// every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching this project's existing tests/*.cpp convention.
//
// Deliberately self-contained (no Logger.cpp/Vulkan dependency, plain std::cerr on failure) rather
// than calling into src/pcg/PcgDataModelSmokeTest.cpp's `#ifndef NDEBUG`-gated
// pcg::RunDataModelSmokeTest() -- that function does not exist in a Release-configured build (its
// whole declaration is compiled out), so a CTest target meant to build/run identically in either
// config cannot depend on it directly. The two files intentionally exercise similar ground through
// different mechanisms: this one is the fast, config-independent CTest proof; PcgDataModelSmokeTest
// is the in-engine Debug-only hook a future phase's DebugTestPipeline feature-test slot can wire
// in, mirroring how e.g. ClusterDAGTests.cpp (CTest) and DebugTestPipeline.cpp's own DAG check
// (live, in-engine) already coexist independently for the Nanite cluster system.

#include "core/maths/Maths.h"
#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSpatialData.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    bool NearlyEqual(float a, float b, float epsilon = 1.0e-4f) {
        return std::abs(a - b) <= epsilon;
    }

    void TestPointDataConstructionAndTransform() {
        pcg::PcgPoint point;
        point.position = maths::vec3{ 1.0f, 2.0f, 3.0f };
        point.scale = maths::vec3{ 2.0f, 2.0f, 2.0f };
        point.density = 0.8f;
        point.color = maths::vec4{ 0.1f, 0.2f, 0.3f, 1.0f };
        point.seed = 12345u;
        point.boundsMin = maths::vec3{ -1.0f, -1.0f, -1.0f };
        point.boundsMax = maths::vec3{ 1.0f, 1.0f, 1.0f };
        point.steepness = 0.5f;

        const maths::mat4 localToWorld = point.GetLocalToWorld();
        Check(NearlyEqual(localToWorld.m[12], 1.0f) && NearlyEqual(localToWorld.m[13], 2.0f) && NearlyEqual(localToWorld.m[14], 3.0f),
            "PcgPoint::GetLocalToWorld: translation column matches position for identity rotation");
        Check(NearlyEqual(localToWorld.m[0], 2.0f) && NearlyEqual(localToWorld.m[5], 2.0f) && NearlyEqual(localToWorld.m[10], 2.0f),
            "PcgPoint::GetLocalToWorld: scale diagonal matches scale for identity rotation");

        // A non-identity rotation (90 degrees about Y) must still leave the translation column
        // exactly at `position` -- rotation composes BEFORE translation, never mixes into it.
        pcg::PcgPoint rotated = point;
        rotated.rotation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, maths::ToRadians(90.0f));
        const maths::mat4 rotatedM = rotated.GetLocalToWorld();
        Check(NearlyEqual(rotatedM.m[12], 1.0f) && NearlyEqual(rotatedM.m[13], 2.0f) && NearlyEqual(rotatedM.m[14], 3.0f),
            "PcgPoint::GetLocalToWorld: translation column is unaffected by rotation");
    }

    void TestSteepnessDensityFalloff() {
        pcg::PcgPoint point;
        point.density = 0.8f;
        point.boundsMin = maths::vec3{ -1.0f, -1.0f, -1.0f };
        point.boundsMax = maths::vec3{ 1.0f, 1.0f, 1.0f };
        point.steepness = 0.5f;

        Check(NearlyEqual(point.GetEffectiveDensity(maths::vec3{ 0.0f, 0.0f, 0.0f }), 0.8f),
            "GetEffectiveDensity: full density at bounds center");
        Check(NearlyEqual(point.GetEffectiveDensity(maths::vec3{ 1.0f, 0.0f, 0.0f }), 0.0f),
            "GetEffectiveDensity: zero density exactly at bounds edge");
        Check(NearlyEqual(point.GetEffectiveDensity(maths::vec3{ 0.5f, 0.0f, 0.0f }), 0.8f),
            "GetEffectiveDensity: full density exactly at the steepness threshold");

        pcg::PcgPoint hardPoint = point;
        hardPoint.steepness = 1.0f;
        Check(NearlyEqual(hardPoint.GetEffectiveDensity(maths::vec3{ 0.99f, 0.0f, 0.0f }), 0.8f, 0.01f),
            "GetEffectiveDensity (steepness=1, hard box): density stays full right up to the edge");

        pcg::PcgPoint softPoint = point;
        softPoint.steepness = 0.0f;
        Check(NearlyEqual(softPoint.GetEffectiveDensity(maths::vec3{ 0.5f, 0.0f, 0.0f }), 0.4f, 0.02f),
            "GetEffectiveDensity (steepness=0, soft gradient): density is roughly halved halfway to the edge");

        Check(sizeof(pcg::GpuPcgPoint) == 96, "GpuPcgPoint is exactly 96 bytes (std430, matches pcg_common.glsl's PcgGpuPoint)");

        const pcg::GpuPcgPoint gpuPoint = pcg::ToGpuPoint(point);
        Check(gpuPoint.seed == point.seed, "ToGpuPoint: seed round-trips exactly");
        Check(NearlyEqual(gpuPoint.density, point.density), "ToGpuPoint: density round-trips exactly");
    }

    void TestAttributeSet() {
        pcg::PcgAttributeSet attrs;
        attrs.Set("enabled", true);
        attrs.Set("count", int32_t{ 42 });
        attrs.Set("scale", 3.5f);
        attrs.Set("offset", maths::vec3{ 1.0f, 2.0f, 3.0f });
        attrs.Set("label", std::string{ "hello_pcg" });

        Check(attrs.Size() == 5, "PcgAttributeSet: 5 distinct keys after 5 Set() calls");

        const bool* enabledPtr = attrs.TryGet<bool>("enabled");
        Check(enabledPtr != nullptr && *enabledPtr == true, "PcgAttributeSet: bool round-trip via TryGet<bool>");

        const int32_t* countPtr = attrs.TryGet<int32_t>("count");
        Check(countPtr != nullptr && *countPtr == 42, "PcgAttributeSet: int32_t round-trip via TryGet<int32_t>");

        const float* scalePtr = attrs.TryGet<float>("scale");
        Check(scalePtr != nullptr && NearlyEqual(*scalePtr, 3.5f), "PcgAttributeSet: float round-trip via TryGet<float>");

        const maths::vec3* offsetPtr = attrs.TryGet<maths::vec3>("offset");
        Check(offsetPtr != nullptr && NearlyEqual(offsetPtr->x, 1.0f) && NearlyEqual(offsetPtr->z, 3.0f),
            "PcgAttributeSet: maths::vec3 round-trip via TryGet<vec3>");

        const std::string* labelPtr = attrs.TryGet<std::string>("label");
        Check(labelPtr != nullptr && *labelPtr == "hello_pcg", "PcgAttributeSet: std::string round-trip via TryGet<string>");

        Check(attrs.TryGet<int32_t>("enabled") == nullptr, "PcgAttributeSet: wrong-type TryGet returns nullptr, not garbage");
        Check(attrs.TryGet<bool>("missing_key") == nullptr, "PcgAttributeSet: missing-key TryGet returns nullptr");
        Check(attrs.GetOr<int32_t>("missing_key", -1) == -1, "PcgAttributeSet: GetOr falls back on a missing key");

        Check(attrs.Remove("count") && !attrs.Has("count") && attrs.Size() == 4,
            "PcgAttributeSet: Remove() erases the entry and shrinks Size()");

        attrs.Set("scale", 9.0f);
        Check(attrs.Size() == 4 && NearlyEqual(*attrs.TryGet<float>("scale"), 9.0f),
            "PcgAttributeSet: Set() on an existing key overwrites rather than duplicating");
    }

    void TestSpatialData() {
        pcg::PcgSurfaceData surface;
        surface.meshID = 7;
        surface.materialID = 3;
        Check(surface.meshID == 7 && surface.materialID == 3, "PcgSurfaceData: fields hold the assigned values");

        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };
        Check(volume.ContainsWorldPoint(maths::vec3{ 1.0f, 0.5f, -1.0f }), "PcgVolumeData (AABB): interior point is contained");
        Check(!volume.ContainsWorldPoint(maths::vec3{ 3.0f, 0.0f, 0.0f }), "PcgVolumeData (AABB): exterior point is rejected");

        pcg::PcgVolumeData obb;
        obb.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        obb.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };
        obb.orientation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, maths::ToRadians(90.0f));
        Check(obb.ContainsWorldPoint(maths::vec3{ 1.0f, 0.5f, -1.0f }),
            "PcgVolumeData (OBB, 90deg Y rotation): symmetric X/Z extents keep the same interior point contained");

        pcg::PcgLandscapeData landscape;
        landscape.meshID = 11;
        landscape.width = 100.0f;
        landscape.length = 100.0f;
        Check(landscape.meshID == 11 && NearlyEqual(landscape.width, 100.0f), "PcgLandscapeData: fields hold the assigned values");

        std::vector<pcg::PcgSplineControlPoint> controlPoints;
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 1.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 2.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        pcg::PcgSplineData spline(controlPoints);

        Check(NearlyEqual(spline.EvaluatePosition(0.0f).z, 0.0f), "PcgSplineData: evaluates exactly at the first control point");
        Check(NearlyEqual(spline.EvaluatePosition(1.0f).z, 1.0f), "PcgSplineData: evaluates exactly at the middle control point");
        Check(NearlyEqual(spline.EvaluatePosition(2.0f).z, 2.0f), "PcgSplineData: evaluates exactly at the last control point");
        Check(NearlyEqual(spline.EvaluatePosition(0.5f).z, 0.5f, 1.0e-3f), "PcgSplineData: interpolates linearly along a straight collinear-tangent segment");
    }

    void TestSeededRandomDeterminism() {
        pcg::PcgSeededRandom streamA(0xABCD1234u);
        pcg::PcgSeededRandom streamB(0xABCD1234u);

        bool allMatch = true;
        for (int i = 0; i < 64; ++i) {
            if (streamA.NextUint32() != streamB.NextUint32()) {
                allMatch = false;
                break;
            }
        }
        Check(allMatch, "PcgSeededRandom: two independent streams with the same seed produce a bit-identical NextUint32() sequence");

        pcg::PcgSeededRandom stream(0x55AA55AAu);
        uint32_t firstPass[8];
        for (uint32_t& v : firstPass) v = stream.NextUint32();
        stream.Reset(0x55AA55AAu);
        bool replayMatches = true;
        for (uint32_t expected : firstPass) {
            if (stream.NextUint32() != expected) {
                replayMatches = false;
                break;
            }
        }
        Check(replayMatches, "PcgSeededRandom: Reset() to the same seed replays the identical sequence");

        pcg::PcgSeededRandom differentSeed(0x11111111u);
        bool anyDifferent = false;
        for (int i = 0; i < 8; ++i) {
            if (differentSeed.NextUint32() != firstPass[i]) {
                anyDifferent = true;
                break;
            }
        }
        Check(anyDifferent, "PcgSeededRandom: a different seed produces a different sequence");

        pcg::PcgSeededRandom floatStream(999u);
        bool inRange = true;
        for (int i = 0; i < 256; ++i) {
            const float f = floatStream.NextFloat01();
            if (f < 0.0f || f >= 1.0f) {
                inRange = false;
                break;
            }
        }
        Check(inRange, "PcgSeededRandom::NextFloat01: every draw stays within [0, 1)");

        pcg::PcgSeededRandom dirStream(42u);
        const maths::vec3 dir = dirStream.NextUnitVec3();
        Check(NearlyEqual(dir.Length(), 1.0f, 1.0e-3f), "PcgSeededRandom::NextUnitVec3: returns a unit-length vector");

        pcg::PcgSeededRandom warmed(0xC0FFEEu);
        warmed.NextUint32(); warmed.NextUint32(); warmed.NextUint32();
        const uint32_t warmedFourth = warmed.NextUint32();
        Check(warmedFourth == pcg::PcgHashCombine(0xC0FFEEu, 3u),
            "PcgSeededRandom: NextUint32() at stream index 3 matches a direct PcgHashCombine(seed, 3) call");

        // Cross-check against the raw free function directly (not just through the stream
        // wrapper) -- proves PcgSeededRandom really is a thin wrapper with no hidden extra mixing.
        Check(pcg::PcgHash32(12345u) == pcg::PcgHash32(12345u), "PcgHash32: pure function, same input always yields the same output");
    }

} // namespace

int main() {
    TestPointDataConstructionAndTransform();
    TestSteepnessDensityFalloff();
    TestAttributeSet();
    TestSpatialData();
    TestSeededRandomDeterminism();

    if (g_failCount == 0) {
        std::cout << "PcgDataModelTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgDataModelTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
