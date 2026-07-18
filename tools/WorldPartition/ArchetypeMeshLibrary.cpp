#include "ArchetypeMeshLibrary.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace worldpartition {

    namespace {

        void PushTri(geometry::SimplifiableMesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
            mesh.triangles.push_back(a);
            mesh.triangles.push_back(b);
            mesh.triangles.push_back(c);
        }

        uint32_t PushVertex(geometry::SimplifiableMesh& mesh, const maths::vec3& position, const maths::vec2& uv) {
            uint32_t index = static_cast<uint32_t>(mesh.positions.size());
            mesh.positions.push_back(position);
            mesh.uvs.push_back(uv);
            mesh.locked.push_back(false);
            return index;
        }

        // ---------------------------------------------------------------------------------------
        // Rock: icosphere (icosahedron, subdivided once, re-projected onto a sphere of `radius`) --
        // mirrors renderer::VulkanContext::GenerateGeometry()'s GenerateIcosphere(0.5f, ...) call for
        // the streaming pool's fine-variant Rock shape (see that block's own case-0 comment).
        // ---------------------------------------------------------------------------------------
        geometry::SimplifiableMesh BuildRockMesh(float radius) {
            geometry::SimplifiableMesh mesh;

            // Base regular icosahedron: 12 vertices at the golden-ratio construction, 20 faces.
            const float t = (1.0f + std::sqrt(5.0f)) * 0.5f;
            std::vector<maths::vec3> baseVerts = {
                {-1.0f,  t, 0.0f}, { 1.0f,  t, 0.0f}, {-1.0f, -t, 0.0f}, { 1.0f, -t, 0.0f},
                { 0.0f, -1.0f,  t}, { 0.0f,  1.0f,  t}, { 0.0f, -1.0f, -t}, { 0.0f,  1.0f, -t},
                {  t, 0.0f, -1.0f}, {  t, 0.0f,  1.0f}, { -t, 0.0f, -1.0f}, { -t, 0.0f,  1.0f},
            };
            std::vector<std::array<uint32_t, 3>> baseFaces = {
                {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
                {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
                {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
                {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1},
            };

            // One subdivision level: each triangle -> 4 (split every edge at its midpoint, then
            // re-normalize every vertex onto the unit sphere before scaling by `radius`) -- 20 * 4 =
            // 80 triangles, dense enough for geometry::SimplifyMeshQEM to have real reduction work
            // to do collapsing down to a small per-cell HLOD triangle budget.
            auto projectAndPush = [&](const maths::vec3& p) -> uint32_t {
                maths::vec3 onSphere = p.Normalize() * radius;
                // Simple equirectangular UV from the unit-sphere direction.
                maths::vec2 uv{
                    0.5f + std::atan2(onSphere.z, onSphere.x) / (2.0f * maths::PI),
                    0.5f - std::asin(std::clamp(onSphere.y / radius, -1.0f, 1.0f)) / maths::PI
                };
                return PushVertex(mesh, onSphere, uv);
            };

            for (const std::array<uint32_t, 3>& face : baseFaces) {
                maths::vec3 a = baseVerts[face[0]];
                maths::vec3 b = baseVerts[face[1]];
                maths::vec3 c = baseVerts[face[2]];
                maths::vec3 ab = (a + b) * 0.5f;
                maths::vec3 bc = (b + c) * 0.5f;
                maths::vec3 ca = (c + a) * 0.5f;

                uint32_t iA = projectAndPush(a);
                uint32_t iB = projectAndPush(b);
                uint32_t iC = projectAndPush(c);
                uint32_t iAB = projectAndPush(ab);
                uint32_t iBC = projectAndPush(bc);
                uint32_t iCA = projectAndPush(ca);

                PushTri(mesh, iA, iAB, iCA);
                PushTri(mesh, iB, iBC, iAB);
                PushTri(mesh, iC, iCA, iBC);
                PushTri(mesh, iAB, iBC, iCA);
            }

            return mesh;
        }

        // ---------------------------------------------------------------------------------------
        // Bush: UV sphere -- mirrors GenerateSphere(0.5f, ...)'s fine-variant Bush shape (case 1).
        // ---------------------------------------------------------------------------------------
        geometry::SimplifiableMesh BuildBushMesh(float radius) {
            geometry::SimplifiableMesh mesh;
            constexpr uint32_t kLatSegments = 8;
            constexpr uint32_t kLonSegments = 12;

            std::vector<std::vector<uint32_t>> ring(kLatSegments + 1, std::vector<uint32_t>(kLonSegments + 1));
            for (uint32_t lat = 0; lat <= kLatSegments; ++lat) {
                float v = static_cast<float>(lat) / static_cast<float>(kLatSegments);
                float theta = v * maths::PI; // 0 (north pole) .. PI (south pole)
                float sinTheta = std::sin(theta);
                float cosTheta = std::cos(theta);

                for (uint32_t lon = 0; lon <= kLonSegments; ++lon) {
                    float u = static_cast<float>(lon) / static_cast<float>(kLonSegments);
                    float phi = u * 2.0f * maths::PI;
                    maths::vec3 position{
                        radius * sinTheta * std::cos(phi),
                        radius * cosTheta,
                        radius * sinTheta * std::sin(phi)
                    };
                    ring[lat][lon] = PushVertex(mesh, position, maths::vec2{ u, v });
                }
            }

            for (uint32_t lat = 0; lat < kLatSegments; ++lat) {
                for (uint32_t lon = 0; lon < kLonSegments; ++lon) {
                    uint32_t a = ring[lat][lon];
                    uint32_t b = ring[lat][lon + 1];
                    uint32_t c = ring[lat + 1][lon];
                    uint32_t d = ring[lat + 1][lon + 1];
                    // Degenerate quads at the poles (a==b's position, c==d's position) collapse to a
                    // zero-area triangle harmlessly -- SimplifyMeshQEM's own fold-over safeguards
                    // treat a zero-area candidate as never worth collapsing into, so it is simply
                    // never selected, not a correctness issue.
                    PushTri(mesh, a, c, b);
                    PushTri(mesh, b, c, d);
                }
            }

            return mesh;
        }

        // ---------------------------------------------------------------------------------------
        // Tree: capsule (cylindrical body + 2 hemispherical caps) -- mirrors GenerateCapsule(0.25f,
        // 0.8f, ...)'s fine-variant Tree shape (case 2): a trunk + canopy silhouette stand-in.
        // ---------------------------------------------------------------------------------------
        geometry::SimplifiableMesh BuildTreeMesh(float radius, float cylinderHeight) {
            geometry::SimplifiableMesh mesh;
            constexpr uint32_t kSegments = 10;
            constexpr uint32_t kCapRings = 4;
            float halfHeight = cylinderHeight * 0.5f;

            // Cylindrical body: 2 rings (top/bottom of the straight section).
            std::vector<uint32_t> topRing(kSegments + 1), bottomRing(kSegments + 1);
            for (uint32_t seg = 0; seg <= kSegments; ++seg) {
                float u = static_cast<float>(seg) / static_cast<float>(kSegments);
                float phi = u * 2.0f * maths::PI;
                float cx = radius * std::cos(phi);
                float cz = radius * std::sin(phi);

                topRing[seg] = PushVertex(mesh, maths::vec3{ cx, halfHeight, cz }, maths::vec2{ u, 0.4f });
                bottomRing[seg] = PushVertex(mesh, maths::vec3{ cx, -halfHeight, cz }, maths::vec2{ u, 0.6f });
            }
            for (uint32_t seg = 0; seg < kSegments; ++seg) {
                PushTri(mesh, bottomRing[seg], bottomRing[seg + 1], topRing[seg]);
                PushTri(mesh, topRing[seg], bottomRing[seg + 1], topRing[seg + 1]);
            }

            // Hemispherical caps: `kCapRings` latitude bands each, sharing the cylinder's own top/
            // bottom ring as their equator so the body and caps stay seamlessly welded.
            auto buildCap = [&](float poleY, float yOffset, bool isTop) {
                std::vector<uint32_t> prevRing = isTop ? topRing : bottomRing;
                for (uint32_t ring = 1; ring <= kCapRings; ++ring) {
                    float v = static_cast<float>(ring) / static_cast<float>(kCapRings);
                    float theta = v * (maths::PI * 0.5f); // 0 (equator) .. PI/2 (pole)
                    float ringRadius = radius * std::cos(theta);
                    float ringY = yOffset + (isTop ? 1.0f : -1.0f) * radius * std::sin(theta);

                    std::vector<uint32_t> curRing(kSegments + 1);
                    if (ring == kCapRings) {
                        uint32_t poleIndex = PushVertex(mesh, maths::vec3{ 0.0f, poleY, 0.0f }, maths::vec2{ 0.5f, isTop ? 0.0f : 1.0f });
                        for (uint32_t seg = 0; seg <= kSegments; ++seg) curRing[seg] = poleIndex;
                    } else {
                        for (uint32_t seg = 0; seg <= kSegments; ++seg) {
                            float u = static_cast<float>(seg) / static_cast<float>(kSegments);
                            float phi = u * 2.0f * maths::PI;
                            maths::vec3 pos{ ringRadius * std::cos(phi), ringY, ringRadius * std::sin(phi) };
                            curRing[seg] = PushVertex(mesh, pos, maths::vec2{ u, isTop ? 0.4f - v * 0.4f : 0.6f + v * 0.4f });
                        }
                    }

                    for (uint32_t seg = 0; seg < kSegments; ++seg) {
                        if (isTop) {
                            PushTri(mesh, prevRing[seg], prevRing[seg + 1], curRing[seg]);
                            if (ring != kCapRings) PushTri(mesh, curRing[seg], prevRing[seg + 1], curRing[seg + 1]);
                        } else {
                            PushTri(mesh, prevRing[seg + 1], prevRing[seg], curRing[seg]);
                            if (ring != kCapRings) PushTri(mesh, curRing[seg + 1], prevRing[seg + 1], curRing[seg]);
                        }
                    }
                    prevRing = curRing;
                }
            };

            buildCap(halfHeight + radius, halfHeight, /*isTop=*/true);
            buildCap(-halfHeight - radius, -halfHeight, /*isTop=*/false);

            return mesh;
        }

        // ---------------------------------------------------------------------------------------
        // Debris: torus -- mirrors GenerateTorus(0.35f, 0.12f, ...)'s fine-variant Debris shape
        // (default case): an irregular scrap-silhouette stand-in.
        // ---------------------------------------------------------------------------------------
        geometry::SimplifiableMesh BuildDebrisMesh(float majorRadius, float minorRadius) {
            geometry::SimplifiableMesh mesh;
            constexpr uint32_t kMajorSegments = 16;
            constexpr uint32_t kMinorSegments = 8;

            std::vector<std::vector<uint32_t>> ring(kMajorSegments + 1, std::vector<uint32_t>(kMinorSegments + 1));
            for (uint32_t majorIdx = 0; majorIdx <= kMajorSegments; ++majorIdx) {
                float u = static_cast<float>(majorIdx) / static_cast<float>(kMajorSegments);
                float majorAngle = u * 2.0f * maths::PI;
                float cosMajor = std::cos(majorAngle);
                float sinMajor = std::sin(majorAngle);

                for (uint32_t minorIdx = 0; minorIdx <= kMinorSegments; ++minorIdx) {
                    float v = static_cast<float>(minorIdx) / static_cast<float>(kMinorSegments);
                    float minorAngle = v * 2.0f * maths::PI;
                    float cosMinor = std::cos(minorAngle);
                    float sinMinor = std::sin(minorAngle);

                    float tubeRadius = majorRadius + minorRadius * cosMinor;
                    maths::vec3 position{
                        tubeRadius * cosMajor,
                        minorRadius * sinMinor,
                        tubeRadius * sinMajor
                    };
                    ring[majorIdx][minorIdx] = PushVertex(mesh, position, maths::vec2{ u, v });
                }
            }

            for (uint32_t majorIdx = 0; majorIdx < kMajorSegments; ++majorIdx) {
                for (uint32_t minorIdx = 0; minorIdx < kMinorSegments; ++minorIdx) {
                    uint32_t a = ring[majorIdx][minorIdx];
                    uint32_t b = ring[majorIdx + 1][minorIdx];
                    uint32_t c = ring[majorIdx][minorIdx + 1];
                    uint32_t d = ring[majorIdx + 1][minorIdx + 1];
                    PushTri(mesh, a, b, c);
                    PushTri(mesh, b, d, c);
                }
            }

            return mesh;
        }

    }

    geometry::SimplifiableMesh BuildArchetypeMesh(uint32_t archetypeShape) {
        switch (archetypeShape % kArchetypeShapeCount) {
            case 0: return BuildRockMesh(0.5f);
            case 1: return BuildBushMesh(0.5f);
            case 2: return BuildTreeMesh(0.25f, 0.8f);
            default: return BuildDebrisMesh(0.35f, 0.12f);
        }
    }

}
