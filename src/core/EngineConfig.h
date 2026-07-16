#pragma once
#include <cstdint>

namespace config {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
// Passe à 0.2f ou 0.25f. Divise par 4 (ou plus) le nombre de sommets générés. 
// Le risque est une perte nette de détails sur les primitives "hero", mais la bande passante de la 3050 l'exige.
constexpr float VERTEX_SPACING = 0.25f;
constexpr float FLOOR_VERTEX_SPACING = 4.0f;

namespace nanite {
// La 3050 possède moins d'ALU. Le rasteriseur logiciel va devenir le goulot d'étranglement. 
// Il faut basculer une plus grande partie de la charge vers le rasteriseur matériel.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 12.0f;
// Force une transition de LOD plus agressive pour limiter la géométrie à traiter.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 2.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

constexpr uint64_t VERTEX_BUFFER_BYTES = 128 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 64 * 1024 * 1024;
} // namespace nanite

namespace lumen {
// Les cœurs RT de la 3050 (Ampere) sont limités. Les mises à jour du cache de surface coûtent cher.
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 1u; // ou 2u maximum
constexpr uint32_t EVICTION_FRAME_DELAY = 120u;

// C'est ici que tu gagnes le plus de temps processeur. 32^3 = 32K probes, c'est intenable. 16^3 = 4K probes.
constexpr uint32_t PROBE_GRID_RESOLUTION = 16u;
constexpr float PROBE_SPACING = 4.0f; // Étend la couverture pour compenser la perte de résolution de la grille.
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 8u; // Moins de rayons tirés par probe.

constexpr uint32_t MAX_TRACED_ENTITIES = 32u;
} // namespace lumen
} // namespace config