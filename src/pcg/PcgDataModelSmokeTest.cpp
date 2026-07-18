// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below) -- see
// PcgDataModelSmokeTest.h's own header comment for the full rationale/scope of this file.
#ifndef NDEBUG

#include "pcg/PcgDataModelSmokeTest.h"

#include "core/Logger.h"
#include "core/maths/Maths.h"
#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSpatialData.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace pcg {

    namespace {

        // Tracks pass/fail across every Check() call in this translation unit, mirroring
        // tests/*.cpp's own `g_failCount`-style accumulator convention (see e.g.
        // tests/BlockCodecTests.cpp) so RunDataModelSmokeTest() can report one final bool.
        int g_FailCount = 0;

        void Check(bool condition, const char* message) {
            if (condition) {
                LOG_INFO(std::string("[PASS] ") + message);
            } else {
                LOG_ERROR(std::string("[FAIL] ") + message);
                ++g_FailCount;
            }
        }

        bool NearlyEqual(float a, float b, float epsilon = 1.0e-4f) {
            return std::abs(a - b) <= epsilon;
        }

        // --- PcgPoint: construction, local-to-world transform, steepness density falloff. -------
        void TestPointData() {
            PcgPoint point;
            point.position = maths::vec3{ 1.0f, 2.0f, 3.0f };
            point.rotation = maths::quat{}; // Identity.
            point.scale = maths::vec3{ 2.0f, 2.0f, 2.0f };
            point.density = 0.8f;
            point.color = maths::vec4{ 0.1f, 0.2f, 0.3f, 1.0f };
            point.seed = 12345u;
            point.boundsMin = maths::vec3{ -1.0f, -1.0f, -1.0f };
            point.boundsMax = maths::vec3{ 1.0f, 1.0f, 1.0f };
            point.steepness = 0.5f;

            const maths::mat4 localToWorld = point.GetLocalToWorld();
            // Identity rotation + uniform scale 2 -> translation column should be exactly `position`,
            // and the scale diagonal should read back exactly 2.0 on each axis.
            Check(NearlyEqual(localToWorld.m[12], 1.0f) && NearlyEqual(localToWorld.m[13], 2.0f) && NearlyEqual(localToWorld.m[14], 3.0f),
                "PcgPoint::GetLocalToWorld: translation column matches position for identity rotation");
            Check(NearlyEqual(localToWorld.m[0], 2.0f) && NearlyEqual(localToWorld.m[5], 2.0f) && NearlyEqual(localToWorld.m[10], 2.0f),
                "PcgPoint::GetLocalToWorld: scale diagonal matches scale for identity rotation");

            // Steepness == 0.5, unit-cube bounds [-1,1]^3 (half-extent 1): center should be full
            // density (edgeFactor 0 <= threshold 0.5), the exact bounds edge should read exactly 0
            // (edgeFactor 1.0 -> falloff = 1 - (1-0.5)/(1-0.5) = 0), and a point exactly at the
            // threshold itself (edgeFactor == steepness) should still read full density.
            const float centerDensity = point.GetEffectiveDensity(maths::vec3{ 0.0f, 0.0f, 0.0f });
            const float edgeDensity = point.GetEffectiveDensity(maths::vec3{ 1.0f, 0.0f, 0.0f });
            const float thresholdDensity = point.GetEffectiveDensity(maths::vec3{ 0.5f, 0.0f, 0.0f });
            Check(NearlyEqual(centerDensity, 0.8f), "PcgPoint::GetEffectiveDensity: full density at bounds center");
            Check(NearlyEqual(edgeDensity, 0.0f), "PcgPoint::GetEffectiveDensity: zero density exactly at bounds edge");
            Check(NearlyEqual(thresholdDensity, 0.8f), "PcgPoint::GetEffectiveDensity: full density exactly at the steepness threshold");

            // steepness == 1.0 (hard box): density should stay full everywhere strictly inside the
            // bounds, even very close to the edge.
            PcgPoint hardPoint = point;
            hardPoint.steepness = 1.0f;
            const float nearEdgeHard = hardPoint.GetEffectiveDensity(maths::vec3{ 0.99f, 0.0f, 0.0f });
            Check(NearlyEqual(nearEdgeHard, 0.8f, 0.01f), "PcgPoint (steepness=1, hard box): density stays full right up to the edge");

            // steepness == 0.0 (soft, linear gradient across the whole bounds): density at the
            // halfway point should be roughly half the intrinsic density.
            PcgPoint softPoint = point;
            softPoint.steepness = 0.0f;
            const float halfwaySoft = softPoint.GetEffectiveDensity(maths::vec3{ 0.5f, 0.0f, 0.0f });
            Check(NearlyEqual(halfwaySoft, 0.4f, 0.02f), "PcgPoint (steepness=0, soft gradient): density is roughly halved halfway to the edge");

            // CPU/GPU mirror struct: ToGpuPoint should round-trip every field exactly.
            const GpuPcgPoint gpuPoint = ToGpuPoint(point);
            Check(NearlyEqual(gpuPoint.positionX, 1.0f) && NearlyEqual(gpuPoint.positionY, 2.0f) && NearlyEqual(gpuPoint.positionZ, 3.0f),
                "ToGpuPoint: position round-trips exactly");
            Check(NearlyEqual(gpuPoint.density, 0.8f), "ToGpuPoint: density round-trips exactly");
            Check(gpuPoint.seed == 12345u, "ToGpuPoint: seed round-trips exactly");
            Check(NearlyEqual(gpuPoint.colorB, 0.3f), "ToGpuPoint: color round-trips exactly");
            Check(sizeof(GpuPcgPoint) == 96, "GpuPcgPoint is exactly 96 bytes (std430, matches pcg_common.glsl's PcgGpuPoint)");
        }

        // --- PcgAttributeSet: typed round-trips for every supported value type. ------------------
        void TestAttributeSet() {
            PcgAttributeSet attrs;
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

            // Wrong-type access must fail gracefully (nullptr), not throw or read garbage.
            Check(attrs.TryGet<int32_t>("enabled") == nullptr, "PcgAttributeSet: wrong-type TryGet returns nullptr, not garbage");
            Check(attrs.TryGet<bool>("missing_key") == nullptr, "PcgAttributeSet: missing-key TryGet returns nullptr");
            Check(attrs.GetOr<int32_t>("missing_key", -1) == -1, "PcgAttributeSet: GetOr falls back on a missing key");

            Check(attrs.Remove("count") && !attrs.Has("count") && attrs.Size() == 4,
                "PcgAttributeSet: Remove() erases the entry and shrinks Size()");

            // Set() on an existing key overwrites in place rather than appending a duplicate.
            attrs.Set("scale", 9.0f);
            Check(attrs.Size() == 4 && NearlyEqual(*attrs.TryGet<float>("scale"), 9.0f),
                "PcgAttributeSet: Set() on an existing key overwrites rather than duplicating");
        }

        // --- PcgSpatialData: Surface/Volume/Landscape/Spline wrappers. ---------------------------
        void TestSpatialData() {
            PcgSurfaceData surface;
            surface.meshID = 7;
            surface.materialID = 3;
            surface.worldOffset = maths::vec3{ 10.0f, 0.0f, -5.0f };
            Check(surface.meshID == 7 && surface.materialID == 3, "PcgSurfaceData: fields hold the assigned values");

            PcgVolumeData volume;
            volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
            volume.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };
            volume.orientation = maths::quat{}; // Identity -> AABB.
            Check(volume.ContainsWorldPoint(maths::vec3{ 1.0f, 0.5f, -1.0f }), "PcgVolumeData (AABB): interior point is contained");
            Check(!volume.ContainsWorldPoint(maths::vec3{ 3.0f, 0.0f, 0.0f }), "PcgVolumeData (AABB): exterior point is rejected");

            // Rotate the same volume 90 degrees around Y: halfExtents.x == halfExtents.z (both 2.0)
            // makes this particular box symmetric under that rotation, so ContainsWorldPoint's
            // world-to-local inverse-rotate step should still classify the same point as interior
            // -- this exercises RotateVector/the conjugate-inverse path end-to-end without needing
            // an asymmetric box to prove a flip (a genuinely different halfExtents.x/z would be
            // needed to observe a flip, which is not what this check is after).
            PcgVolumeData obb;
            obb.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
            obb.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };
            obb.orientation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, maths::ToRadians(90.0f));
            Check(obb.ContainsWorldPoint(maths::vec3{ 1.0f, 0.5f, -1.0f }), "PcgVolumeData (OBB, 90deg Y rotation): symmetric extents keep the same interior point contained");

            PcgLandscapeData landscape;
            landscape.meshID = 11;
            landscape.worldOffset = maths::vec3{ 0.0f, -0.8f, 0.0f };
            landscape.width = 100.0f;
            landscape.length = 100.0f;
            Check(landscape.meshID == 11 && NearlyEqual(landscape.width, 100.0f), "PcgLandscapeData: fields hold the assigned values");

            // Spline: a straight line (tangents along +Z, matching spline_deformation.glsl's own
            // "reference axis" convention) should evaluate exactly at each control point for an
            // integer parameter, and linearly halfway between them for the midpoint parameter.
            std::vector<PcgSplineControlPoint> controlPoints;
            controlPoints.push_back(PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            controlPoints.push_back(PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 1.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            controlPoints.push_back(PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 2.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            PcgSplineData spline(controlPoints);

            const maths::vec3 atStart = spline.EvaluatePosition(0.0f);
            const maths::vec3 atMid = spline.EvaluatePosition(1.0f);
            const maths::vec3 atEnd = spline.EvaluatePosition(2.0f);
            const maths::vec3 atQuarter = spline.EvaluatePosition(0.5f);
            Check(NearlyEqual(atStart.z, 0.0f), "PcgSplineData: evaluates exactly at the first control point");
            Check(NearlyEqual(atMid.z, 1.0f), "PcgSplineData: evaluates exactly at the middle control point");
            Check(NearlyEqual(atEnd.z, 2.0f), "PcgSplineData: evaluates exactly at the last control point");
            Check(NearlyEqual(atQuarter.z, 0.5f, 1.0e-3f), "PcgSplineData: interpolates linearly along a straight collinear-tangent segment");
        }

        // --- PcgSeededRandom: bit-identical determinism given the same seed. ---------------------
        void TestSeededRandomDeterminism() {
            PcgSeededRandom streamA(0xABCD1234u);
            PcgSeededRandom streamB(0xABCD1234u); // Same seed, independent instance.

            bool allMatch = true;
            for (int i = 0; i < 64; ++i) {
                const uint32_t a = streamA.NextUint32();
                const uint32_t b = streamB.NextUint32();
                if (a != b) {
                    allMatch = false;
                    break;
                }
            }
            Check(allMatch, "PcgSeededRandom: two independent streams with the same seed produce a bit-identical NextUint32() sequence");

            // Reset() back to the original seed must reproduce the exact same sequence again.
            PcgSeededRandom stream(0x55AA55AAu);
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

            // A different seed must (overwhelmingly likely) diverge from the first sequence.
            PcgSeededRandom differentSeed(0x11111111u);
            bool anyDifferent = false;
            for (int i = 0; i < 8; ++i) {
                if (differentSeed.NextUint32() != firstPass[i]) {
                    anyDifferent = true;
                    break;
                }
            }
            Check(anyDifferent, "PcgSeededRandom: a different seed produces a different sequence");

            // NextFloat01() must stay within its documented [0,1) range across many draws.
            PcgSeededRandom floatStream(999u);
            bool inRange = true;
            for (int i = 0; i < 256; ++i) {
                const float f = floatStream.NextFloat01();
                if (f < 0.0f || f >= 1.0f) {
                    inRange = false;
                    break;
                }
            }
            Check(inRange, "PcgSeededRandom::NextFloat01: every draw stays within [0, 1)");

            // NextUnitVec3() must stay unit-length (within float tolerance).
            PcgSeededRandom dirStream(42u);
            const maths::vec3 dir = dirStream.NextUnitVec3();
            Check(NearlyEqual(dir.Length(), 1.0f, 1.0e-3f), "PcgSeededRandom::NextUnitVec3: returns a unit-length vector");

            // Two DIFFERENT PcgSeededRandom instances constructed with the SAME seed but consuming
            // a DIFFERENT number of values first must still land on the same value at the same
            // stream index once realigned -- proves the hash is genuinely a pure function of
            // (seed, index), not incidentally dependent on call history.
            PcgSeededRandom warmed(0xC0FFEEu);
            warmed.NextUint32(); warmed.NextUint32(); warmed.NextUint32(); // Consume 3 values.
            const uint32_t warmedFourth = warmed.NextUint32();            // 4th value (index 3).
            Check(warmedFourth == PcgHashCombine(0xC0FFEEu, 3u),
                "PcgSeededRandom: NextUint32() at stream index 3 matches a direct PcgHashCombine(seed, 3) call");
        }

    } // namespace

    bool RunDataModelSmokeTest() {
        g_FailCount = 0;
        LOG_INFO("[PcgDataModelSmokeTest] Starting PCG Phase 1 data-model smoke test...");

        TestPointData();
        TestAttributeSet();
        TestSpatialData();
        TestSeededRandomDeterminism();

        if (g_FailCount == 0) {
            LOG_INFO("[PcgDataModelSmokeTest] All checks passed.");
        } else {
            LOG_ERROR(std::string("[PcgDataModelSmokeTest] ") + std::to_string(g_FailCount) + " check(s) FAILED.");
        }
        return g_FailCount == 0;
    }

}

#endif // NDEBUG
