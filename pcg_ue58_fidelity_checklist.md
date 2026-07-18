# PCG Framework: UE5.8 Fidelity Checklist

**PCG roadmap Phase 9.1 ("Checklist comparative face à UE5.8")** — a code-verified comparison between
this engine's from-scratch PCG (Procedural Content Generation) framework (`src/pcg/`, plus its
renderer/world/tools integration points) and Unreal Engine 5.8's real, shipping PCG plugin.

## Methodology

Every row below was verified by **reading the actual implementation**, not by trusting a phase's own
roadmap description. For anything claiming "this is wired into a real pipeline," the specific call site
was located with `grep` across `src/`, `tools/`, and `CMakeLists.txt` (excluding `tests/`) — the same
"is this actually invoked, or only present" check this project's own history has repeatedly needed (see
`MEMORY.md`'s `project_pcg_smoke_test_build_corruption_confirmed` entry for a prior instance of exactly
this class of question). UE5.8 ground truth was pulled from Epic's own current documentation rather than
relied on from training-data recollection alone — see **Sources** at the bottom.

Status legend:
- **Full parity** — equivalent capability exists, is real (not a stub), and is reachable from actual
  running code (test harness and/or the live application).
- **Full parity (adapted)** — the capability exists but necessarily takes a different concrete shape
  because this engine has no Blueprint/UObject/reflection VM (see `CLAUDE.md`) — a deliberate adaptation,
  not a compromise.
- **Partial** — a real, working piece exists but with a materially narrower scope than UE5.8's version,
  and/or it is not actually connected end-to-end to the rest of the pipeline yet.
- **Gap** — no equivalent exists. Each Gap row states whether it is small enough that this pass could
  have fixed it (none were) or whether it is real future-phase-sized work (see **Named Gaps** section).

---

## A. Core Data Model & Determinism

| UE5.8 capability | This engine's equivalent | Status | Notes |
|---|---|---|---|
| `FPCGPoint` (transform, density, color, seed, bounds, steepness) | `pcg::PcgPoint` / `pcg::GpuPcgPoint` — `src/pcg/PcgPointData.h` | **Full parity** | CPU and GPU-mirror structs, `static_assert`-checked byte layout against `pcg_common.glsl`'s `PcgGpuPoint`. Steepness-driven edge falloff (`ComputeDensityFalloff`) is a documented best-effort reproduction of UE5.8's semantics (Epic's own source isn't available to cross-check bit-for-bit). |
| Attribute Sets / Params Data | `pcg::PcgAttributeSet` — `src/pcg/PcgAttributeSet.h` | **Full parity (adapted)** | Closed `std::variant<bool,int32_t,float,vec3,string>` vs. UE5.8's open, reflection-based `FPCGMetadata` attribute system. Sufficient for this engine's node-param use case; narrower type set (no per-point custom attributes beyond the fixed `PcgPoint` fields — see Gap in section C). |
| Deterministic seeding (`FPCGPoint::Seed`, per-node Seed override) | `pcg::PcgSeededRandom`, `PcgHash32`/`PcgHashCombine` — `src/pcg/PcgSeededRandom.h` | **Full parity (exceeds)** | Stateless hash-based (not a stream PRNG), CPU/GPU bit-identical by construction and empirically cross-validated (see Terrain Sampler row). Every sampler/filter/spawner/biome file in this audit documents and follows a strict per-derivation seed-salting discipline. This is a stronger, more explicitly-enforced guarantee than UE5.8's own public docs specify. |

## B. Spatial Data Sources & Samplers

| UE5.8 node | This engine's equivalent | Status | Notes |
|---|---|---|---|
| **Surface Sampler** | `pcg::SampleSurfacePoints` — `src/pcg/PcgSurfaceSampler.h/.cpp` | **Partial** | Algorithm complete (area-weighted triangle selection, barycentric placement, tangent-plane jitter) and unit-tested (`tests/PcgSurfaceSamplerTests.cpp`, 296 lines). **Zero production call sites** (grep-verified across `src/` and `tools/`) — reachable only from its own CTest. It deliberately takes an explicit CPU-side triangle list rather than resolving real Nanite cluster geometry (reading GPU-resident cluster pages back to the CPU is explicitly out of scope, per the header's own "SCOPE BOUNDARY" comment) — so it cannot sample this engine's actual procedural meshes yet even if something called it. |
| **Volume Sampler** | `pcg::SampleVolume` — `src/pcg/PcgVolumeSampler.h/.cpp` | **Full parity** | Grid + Random modes, correct OBB handling (rotation-aware). This is the **only** sampler with a real end-to-end call site: `ClusterRenderPipeline::RunPcgFullPipelineSmokeTest` (`src/renderer/ClusterRenderPipeline.cpp:3902`) drives Volume Sampler → Self-Pruning → Mesh Spawner → `PcgInstanceDrawPass` and actually rasterizes the result. |
| **Get Landscape Data** / landscape sampling | `pcg::SampleTerrainPoints` + CPU port of the terrain heightfield (`SampleTerrainHeightLocalCPU`) — `src/pcg/PcgTerrainSampler.h/.cpp` | **Partial** | Notably strong in isolation: the CPU port was cross-checked against the **real GPU shader** via a throwaway compute dispatch on actual hardware (RTX 5080 Laptop), catching and fixing a real negative-coordinate rounding bug (documented in the file's own header). Matches to ~5e-8 float32 noise. But — same as Surface Sampler — **zero production call sites**; only `tests/PcgTerrainSamplerTests.cpp` exercises it. |
| **Spline Sampler** | `pcg::SampleSplineByArcLength` — `src/pcg/PcgSplineSampler.h/.cpp` | **Partial** | True arc-length (not raw-`t`) placement, correctly avoids the naive uniform-`t` bunching bug. **Zero production call sites** outside `tests/PcgSplineSamplerTests.cpp`. |
| **Texture Sampler** / **Get Texture Data** (splatmap/mask-driven density) | — | **Gap** | No texture-driven sampling or density input exists anywhere in `src/pcg/`. Confirmed via grep (`texture` only appears in a `PcgGraph.h` comment about a *future* pin data type). Real future-phase work, not a small fix — needs a texture-read path threaded into the PCG data model. |
| **Mesh Sampler** (points from an existing mesh's vertices) | — | **Gap** | No equivalent; only surface-area sampling exists (a different sampling basis). |

**Sampler summary**: all 4 core samplers this engine's roadmap targeted (Surface/Terrain/Volume/Spline)
are algorithmically real and individually well-tested — but only the Volume Sampler is actually exercised
by anything the application runs. Surface/Terrain/Spline are, today, tested library code with no caller.

## C. Density / Filter Nodes

| UE5.8 node | This engine's equivalent | Status | Notes |
|---|---|---|---|
| **Density Filter** | `pcg::FilterByDensity` — `src/pcg/PcgDensityTransformFilter.h/.cpp` | **Full parity** | Filters on raw `density`, matching UE5.8's own documented behavior (not the steepness-adjusted effective density). |
| **Density Remap** | `pcg::RemapDensity` — same file | **Full parity** | Linear remap, degenerate-range-safe. |
| **Attribute Noise** / **Density Noise** | `pcg::SampleAttributeNoise` / `pcg::ApplyNoiseToDensity` — same file | **Full parity** | Coherent 3D value noise (trilinear + smoothstep), same conceptual pairing as UE5.8's Metadata-category "Attribute Noise"/"Density Noise" nodes. |
| **Transform Points** | `pcg::ApplyTransformJitter` — same file | **Full parity** | Independently toggleable position/rotation/scale jitter with a fixed, documented per-point draw order. |
| **Self Pruning** | `pcg::PruneByDistance` + `pcg::PcgSpatialHashGrid` — `src/pcg/PcgSelfPruningFilter.h/.cpp` | **Full parity** | Exact UE5.8 node-name match. Uniform hash grid (cell = 2×minDistance), O(1) amortized per point, deterministic earlier-index-wins rule. Both `Uniform` and `ScaledByBounds` modes. |
| **Union** | `pcg::Union` — `src/pcg/PcgBooleanSetOps.h/.cpp` | **Full parity** | Non-deduplicating concatenation, matching UE5.8's own point-data Union behavior. |
| **Difference / Intersection** | `pcg::IntersectWithVolume`, `pcg::DifferenceFromVolume`, `pcg::DifferenceFromSpline`, `pcg::IntersectMultipleVolumes` — same file | **Partial** | UE5.8's `Difference`/`Intersection`/`Inner Intersection` nodes operate generically over *any* Spatial Data pair (point-vs-point, point-vs-volume, point-vs-spline, ...). This engine's versions are narrower: point-set-vs-**Volume** or point-set-vs-**Spline** only — no general point-set-vs-point-set boolean op. This is a deliberate, documented scoping decision matching the common real-world usage pattern ("no trees within N meters of the road"), not an oversight, but it is narrower than the UE5.8 node family. |
| **Projection** | `pcg::ProjectOntoTerrain` — `src/pcg/PcgSlopeHeightFilter.h/.cpp` | **Full parity** | Re-samples terrain height/normal at each point's current (x,z) and re-aligns — matches UE5.8 Projection's re-snap-to-surface semantics, scoped specifically to this engine's terrain (UE5.8's Projection can target arbitrary spatial data). |
| **Discard Points on Irregular Surface** / Slope Filter | `pcg::FilterBySlope` (2 overloads) — same file | **Full parity** | Real ground-truth overload (reads the Terrain Sampler's actual analytic slope) plus an approximate rotation-derived overload for non-terrain point sets, both clearly documented as to which is which. |
| Height-range filtering | `pcg::FilterByHeight` — same file | **Full parity** | No single dedicated UE5.8 "Height Filter" node exists as such (achieved via generic Attribute/Point Filter Range on the Z/height attribute in 5.8) — this engine provides it as first-class, a reasonable ergonomic addition. |
| Per-point custom/arbitrary attributes (UE5.8's `FPCGMetadata`, the ~80 generic Attribute Maths/Boolean/Trig/Vector/Compare-Op nodes) | — | **Gap** | UE5.8's PCG graph is also a general visual-scripting environment over arbitrary per-point metadata (add/rename/transfer/math arbitrary attributes). This engine's `PcgAttributeSet` supports arbitrary keys at the *node-param* level but `PcgPoint` itself has no arbitrary-attribute bag, and none of UE5.8's generic Attribute-Op node categories (Maths/Boolean/Trig/Vector/Compare/Rotator/Transform Op, ~80 node names total) have any equivalent here. Real, large future work — not attempted. |

## D. Spawner Nodes

| UE5.8 node | This engine's equivalent | Status | Notes |
|---|---|---|---|
| **Static Mesh Spawner** (weighted mesh selection) | `pcg::SpawnFromPoints` — `src/pcg/PcgMeshSpawner.h/.cpp` | **Full parity (exceeds)** | Cumulative-weight selection (same technique as the Surface Sampler's own triangle selection). Wired all the way to actual GPU-instanced rendering of real Nanite cluster geometry via `pcg::PcgInstanceSpawnManager` (`src/pcg/PcgInstanceSpawnManager.h`) → `renderer::PcgInstanceDrawPass` (`src/renderer/passes/PcgInstanceDrawPass.h/.cpp`), which reuses the engine's real `ClusterCullingPass` GPU frustum/backface cull + indirect draw path. This is genuinely more sophisticated than a literal "spawn a static mesh actor" node needs to be, because it plugs into this engine's own virtualized-geometry pipeline. |
| **Spline Mesh Spawner** (deform a mesh segment-by-segment along a spline — fences, pipes, guardrails) | — | **Gap** | No equivalent. The Spline Sampler places rigid point instances at intervals along a spline (works for fence *posts*), but nothing deforms/bends a mesh's own geometry to follow the curve continuously. Real future work, not attempted here. |
| **Spawn Actor** / **Create Target Actor** (spawn a full game-logic actor, not just a mesh) | — | **N/A** | This engine has no actor/gameplay-object system to spawn into (matches the project's "100% procedural GPU-driven visual demo, not a game" scope) — not a meaningful gap for this project. |

## E. PCG Graph Engine

| UE5.8 capability | This engine's equivalent | Status | Notes |
|---|---|---|---|
| Node-based DAG, typed pins | `pcg::PcgGraph` — `src/pcg/PcgGraph.h/.cpp` | **Full parity** | Real DAG: `AddLink` validates pin existence, type compatibility, single-connection-per-input, and acyclicity *before* mutating anything (`WouldCreateCycle` rejected outright, never silently accepted). Closed `PcgPinData` variant (Points/AttributeSet/Surface/Volume/Landscape/Spline/SpawnRequests). |
| Subgraphs | `PcgNode::subgraph` + `SubgraphPinBinding`, evaluated by `PcgGraphEvaluator` | **Full parity** | Nested graph support with explicit external-seed pin binding. |
| Graph serialization / save-load | `PcgGraph::SerializeToJson` / `DeserializeFromJson` — `src/pcg/PcgGraph.cpp` | **Full parity** | Human-inspectable JSON via vendored `crude_json`; round-trip structural-equality tested. |
| CPU graph evaluation | `pcg::PcgGraphEvaluator` — `src/pcg/PcgGraphEvaluator.h/.cpp` | **Full parity (narrower execution model)** | Kahn's-algorithm topological sort, per-node output caching. UE5.8 5.8 specifically added parallel per-node-job DAG evaluation (independent nodes run concurrently across worker threads, ~2-2.5x faster full regen per Epic's own 5.8 notes); this engine's evaluator is a straightforward single-threaded walk. Reasonable for this project's bounded, bake-oriented graphs, but a real difference worth naming — may be worth revisiting alongside Phase 9.3's profiling/budget work. |
| Native C++ node authoring (`UPCGSettings`/`FPCGElement` subclassing) | `PCG_REGISTER_NODE_TYPE` macro + `PcgNodeExecuteFn` — `src/pcg/PcgNodePlugin.h/.cpp` | **Full parity (adapted)** | No Blueprint/UObject reflection exists in this engine (by design, see `CLAUDE.md`), so this is necessarily a different mechanism — a self-registering macro pairing a declarative pin schema with a `std::function` execute callback — but it delivers the same practical capability: author a new node type in native code with an introspectable schema. `PcgNodeTypeCatalog` + `ValidateGraphAgainstCatalog` give whole-graph pre-evaluation validation UE5.8 also provides. |
| GPU-resident/compute-driven node execution | `PcgGpuNodeExecuteFn`, `PcgGraphEvaluator::EvaluateNodeGpu`, reference implementation `pcg::PcgGpuDensityNoiseNode` — `src/pcg/PcgGraphEvaluator.h`, `PcgGpuDensityNoiseNode.h/.cpp` | **Full parity (adapted, exceeds)** | UE5.8's PCG graph runs on the CPU; per-point compute is not GPU-dispatched by the graph engine itself. This engine proves a genuinely additive capability (record compute dispatches directly from a graph node, no CPU readback) that UE5.8 does not offer at this level — appropriate for this project's "100% procedural GPU driven" mandate. Scoped deliberately to a single proof-of-concept node type; the topological evaluator does not yet walk a mixed CPU/GPU graph automatically (documented, explicit scope limit in `PcgGraphEvaluator.h`). |
| **Registered graph node types available for actual authoring** | `pcg.spawner.weighted_mesh` (`src/pcg/PcgMeshSpawner.cpp:145`) | **Gap (significant)** | This is the single largest structural gap found in this audit. Grepping every `PCG_REGISTER_NODE_TYPE(...)` call site across `src/` (excluding `tests/`) turns up exactly **one** production node type — the weighted mesh spawner. None of the 4 samplers (section B) or the filter primitives (section C) are registered as `PcgGraph` node types; they exist only as directly-callable C++ functions. `PcgCellGenerator.h`'s own header comment already flags this ("Phase 2's samplers/Phase 3's filters are not yet wired into the graph-node registry"). **Practical consequence**: an actual `pcg::PcgGraph` today cannot express "Surface Sampler → Density Filter → Self-Pruning → Mesh Spawner" as a connected node graph — that composition exists only as hand-written C++ call sequences (e.g. the smoke test in `ClusterRenderPipeline.cpp`). Not a small fix: Surface Sampler's node-registration is blocked on the still-unsolved "read Nanite geometry back to CPU" problem it explicitly defers; the others would need a real param-encoding design per node type (`PcgAttributeSet`'s closed variant has no `quat` alternative, so even the comparatively simple Volume Sampler's `orientation` field has no obvious encoding without a real design decision). Documented here as real, named future work — see **Named Gaps**. |

## F. PCG Volumes & World Partition Runtime Generation

| UE5.8 capability | This engine's equivalent | Status | Notes |
|---|---|---|---|
| PCG Volume (authorable bounds + graph reference) | `worldpartition::PcgVolumeDesc` / `BuildPcgVolumeActorRecord` / `TryParsePcgVolumeDesc` — `tools/WorldPartition/PcgVolumeActor.h/.cpp` | **Full parity** | Real OFPA-style actor record (bounds + graph asset path reference + seed), round-trips through the same generic actor-file pipeline as every other actor type. |
| Cell-based spatial bucketing of PCG Volumes | `world::PcgVolumeCellIndex` / `ScanPcgVolumeActorFiles` / `BuildPcgVolumeCellIndex` — `src/world/PcgVolumeCellIndex.h/.cpp` | **Full parity** | |
| Partitioned per-cell generation (content clipped to the cell it's generated for, not duplicated) | `pcg::GeneratePcgContentForCell` — `src/pcg/PcgCellGenerator.h/.cpp` | **Full parity** | Correctly clips a volume's terminal graph output to each overlapping cell's own bounds via `IntersectWithVolume`/position tests — genuinely partitioned, not "every cell gets the whole volume's content." |
| Live runtime generation triggered by streaming (UE5.8: `UPCGComponent` auto-generates as an actor streams in) | `world::PcgCellLoader` (`IWorldCellLoader` implementation) — `src/world/PcgCellLoader.h/.cpp` | **Partial (Debug-only)** | Structurally correct: worker-thread `LoadCellFullDetail`/`UnloadCell` stage events, main-thread `Pump()` drains them and calls `PcgInstanceSpawnManager::SpawnInstances`/`DespawnInstances` — genuinely wired to `world::IWorldCellLoader`'s real threading contract, verified end-to-end by `ClusterRenderPipeline::RunPcgCellLoaderSmokeTest` (called from `main.cpp` at real startup, not just a CTest). **However, the entire file is wrapped in `#ifndef NDEBUG`** — it produces zero object code in Release. This is a deliberate, well-justified decision (it depends on `tools/WorldPartition/`'s OFPA actor parser, itself Debug-only by established project convention — see `WorldPartitionTypes.h`), explicitly acknowledged in the code's own header comment as a scope limit for a future phase to lift. **Practical consequence**: in a shipping Release build, an authored PCG Volume cannot generate any world content at runtime at all — the whole World-Partition-driven procedural population feature does not exist outside Debug. |
| Generation-result caching (avoid re-evaluating a graph every time a cell reloads) | `PcgCellLoader::m_GenerationResultCache`, Phase 6.4 | **Full parity** | Coord-keyed memoization, correctly never evicted by `UnloadCell` (generation is a pure function of coord for a given loader instance — see the class's own "Phase 6.4" comment for the bound proof). |
| Bake-vs-runtime determinism validation | `RunPcgCellLoaderSmokeTest`'s own STEP 5 (Phase 6.5), `GetCachedResultForTest` | **Full parity** | Directly compares the live runtime path's cached output against a separately-computed "offline bake" call for the same input, proving no silent divergence. This is *more* rigorous than what UE5.8's public docs describe. |
| HLOD-of-PCG (collapse many PCG-scattered instances into one simplified proxy for distant cells) | `HlodPipeline::BuildHlodForPcgScatter` / `GatherPcgScatterMeshes` — `tools/WorldPartition/HlodPipeline.h/.cpp` | **Partial** | The collapse *algorithm* is real (feeds scattered `PcgPoint`s + per-point archetype shapes into the existing mesh-simplification backend, correctly bounded by triangle budget) and unit-tested (`tests/HlodPipelineTests.cpp`). But — grep-verified — it is only ever called from its own synthetic demo generator (`BuildHlodForSyntheticPcgScatterDemo`) and its own tests; it is **never** fed by the real Phase 6 PCG-Volume-driven generation pipeline. Separately, the *live* runtime path has no HLOD tier at all: `PcgCellLoader::LoadCellHlod` is an explicit, logged no-op (PCG content only ever appears at full detail or not at all). Two real, independently-documented gaps versus UE5.8's actual HLOD-layer auto-assignment for generated content. |
| Graph-level exposed "Graph Parameters" (a placed instance overrides a named value without editing the graph's internal wiring) | — | **Gap (explicitly declined, documented)** | `PcgVolumeInspector.h`'s own header comment discusses this directly: it names the full design (a new exposed-parameter declaration on `PcgGraph`, a JSON schema bump, a per-volume override map, evaluator-side plumbing) and explicitly scopes it out as "a multi-file, multi-phase design change, not a quick addition," delivering only per-volume field editing (seed/bounds/graph path) instead. Confirmed still accurate — not implemented anywhere in this codebase. |

## G. PCG Biomes / Ecosystem

| UE5.8 capability | This engine's equivalent | Status | Notes |
|---|---|---|---|
| Biome asset: named, ordered layer stack (e.g. Grass/Rocks/Trees), each independently filtering a shared base point set | `pcg::PcgBiomeRuleSet` / `pcg::ApplyBiome` — `src/pcg/PcgBiomeRules.h/.cpp` | **Full parity** | Composition-only over existing Phase 3 filter primitives (density weight/filter, height/slope filter, exclusion volumes, noise variation, self-pruning), each layer independently seeded (`PcgHashCombine(seed, layerIndex)`) so reordering/adding layers never perturbs an unrelated layer's output. |
| Cross-layer ecosystem rules (e.g. "no bushes under a tree's canopy") | `pcg::ApplyEcosystemExclusion` / `pcg::ApplyEcosystemRules` — `src/pcg/PcgEcosystemExclusion.h/.cpp` | **Full parity** | Reuses the Self-Pruning filter's spatial hash grid for a proper two-set (suppressing-vs-suppressed) proximity query; supports both a flat radius and a per-point bounds-derived radius (e.g. a big tree suppresses a wider area than a small one). Rules compose in list order over a shared, mutating result table — documented, deterministic. |
| Climate-driven biome selection (temperature/moisture-eligible biome variants) | `pcg::SelectAndApplyClimateBiome` / `pcg::SampleCurrentClimate` — `src/pcg/PcgClimateBiomeSelector.h/.cpp` | **Full parity** | Genuinely reads **live** engine state (`renderer::AtmosClimatePass::GetEffectiveTemperatureCelsius()` / the newly-added `GetEffectiveRelativeHumidity()`, both verified present and real, not stubs), not synthetic climate data. Soft per-axis suitability falloff (not a hard cutoff), matching UE5.8's documented "climate suitability curve" behavior; an all-candidates-ineligible climate correctly yields zero biome selected rather than an arbitrary best-of-a-bad-lot pick. |

## H. Editor Tooling

| UE5.8 capability | This engine's equivalent | Status | Notes |
|---|---|---|---|
| PCG Graph editor (drag nodes from a palette, wire pins, live-updates the viewport) | `renderer::debug::PcgGraphEditorPanel` — `src/renderer/debug/PcgGraphEditorPanel.h/.cpp` | **Gap (structural, documented as a scaffold by its own header)** | The panel's own header comment states plainly: "a PURE EDITOR SCAFFOLD... No real PCG graph data model exists yet." Confirmed still true: `Draw()` takes no `pcg::PcgGraph` parameter at all, and `main.cpp`'s own call-site comment says the two demo nodes it spawns "are still wired to nothing." A developer can drag/connect nodes on this canvas, but nothing about that canvas is a real, executable `pcg::PcgGraph` — building an actual graph today requires calling `PcgGraph::AddNode`/`AddLink` directly in C++ or hand-writing its JSON. This is Debug-only (`#ifndef NDEBUG`), matching CLAUDE.md's tooling-exclusion rule, which is correct regardless of the scaffold's own completeness. |
| Live point-cloud / gizmo debug visualization | `renderer::debug::PcgPointCloudDebugView` — `src/renderer/debug/PcgPointCloudDebugView.h/.cpp` | **Full parity** | Real wireframe-AABB-per-point 3D gizmo draw (density → red/green color), correctly depth-tested. Fed from `RunPcgFullPipelineSmokeTest`'s actual sampled/filtered point set — genuine data, not synthetic — though it is a one-shot upload of that one smoke test's result, not a live re-visualization of an arbitrary/currently-edited graph. |
| Per-node data inspector (see what data actually flowed through a node's pins) | `renderer::debug::PcgNodeDataInspector` — `src/renderer/debug/PcgNodeDataInspector.h/.cpp` | **Partial** | The inspector logic itself is real and generic (`Draw(graph, evalResult, catalog)` will faithfully show any `pcg::PcgGraph`'s evaluated pin data, exhaustive `std::visit`-based summarization). But as actually wired in `main.cpp`, it is only ever handed `g_PcgInspectorDemoGraph` — its own small, self-built, hardcoded 4-node demo graph (`BuildDemoInspectorGraph`), **not** whatever is on the Graph Editor Panel's canvas (which has no real graph to show anyway, see above) and not any real authored PCG Volume's graph. Three of the four Phase 7.x panels each construct their own independent demo/fallback dataset rather than sharing one real live graph — worth knowing before assuming the "PCG Graph Editor" tab is one integrated tool. |
| PCG Volume authoring/override UI | `renderer::debug::PcgVolumeInspector` — `src/renderer/debug/PcgVolumeInspector.h/.cpp` | **Partial** | Real scan-and-edit loop (seed/bounds/graph-asset-path, in-memory) against actual `.actor` files when present. Falls back to 3 synthetic in-memory volumes today because — accurately, per its own comment — nothing in this codebase yet authors a real `PcgVolume` actor file (`BakeDemoWorld.cpp` only emits Rock/Bush/Tree/Debris). **No write-back**: edits never reach disk (`BuildPcgVolumeActorRecord` + `WriteActorFile` is explicitly not called by the "Save" button, which only logs where it would have written) — explicitly out of scope per the file's own header, not an oversight. |

---

## Named Gaps (real future-phase work — not attempted in this pass, per 9.1's scope discipline)

Ordered roughly by how much they'd change what an author can actually *do* with this PCG framework:

1. **Register Phase 2 samplers + Phase 3 filters as real `PcgGraph` node types** (section E). The single
   biggest gap: today only the weighted mesh spawner is a real graph node. Blocked in part on Surface
   Sampler's own unsolved "resolve real Nanite geometry to a CPU triangle list" scope boundary; the
   others need a genuine `PcgAttributeSet` param-encoding design per node type (no `quat` alternative
   exists for e.g. Volume Sampler's `orientation`).
2. **Wire the PCG Graph Editor canvas to a real `pcg::PcgGraph`** (section H) — today it is an
   acknowledged UI scaffold. Meaningfully depends on gap 1 (there'd be little to drag onto the canvas
   otherwise).
3. **Promote World Partition PCG Volume runtime generation to Release** (section F) — `world::PcgCellLoader`
   and its `tools/WorldPartition/` OFPA dependency chain are Debug-only by current project convention.
   A real architectural call (how much of the offline actor-parsing toolchain becomes Release-safe),
   already flagged as such by the code's own comments.
4. **Texture-driven PCG sampling/density** (section B) — no equivalent to UE5.8's Texture Sampler /
   Get Texture Data anywhere in this engine's PCG.
5. **Spline Mesh Spawner** (section D) — continuous mesh deformation along a spline (fences/pipes),
   distinct from the existing point-at-intervals spawning.
6. **HLOD-of-PCG end-to-end wiring** (section F) — `BuildHlodForPcgScatter`'s algorithm is real but
   unreachable from the actual Phase 6 generation pipeline; the live runtime path has no HLOD tier at all.
7. **Graph-level exposed parameters/overrides** (section F) — explicitly declined by Phase 7.4's own
   scope note; still accurate.
8. **Arbitrary per-point metadata + generic Attribute-Op node families** (section C) — UE5.8's PCG graph
   doubles as a general visual-scripting environment (~80 generic Maths/Boolean/Trig/Vector/Compare-Op
   node names); this engine implements none of that generic layer, only domain-specific PCG nodes.
9. **Parallel per-node-job graph evaluation** (section E) — UE 5.8 specifically added this
   (~2-2.5x full-regen speedup per Epic's own 5.8 notes); this engine's evaluator is single-threaded.
   Possibly worth coordinating with Phase 9.3's profiling/budget work rather than treating as fully
   separate.

No item above was small enough to fix within this pass's scope discipline — each either requires a real
design decision (param encoding, an exposed-parameter schema) or touches a deliberate, already-justified
architectural boundary (the Release/Debug linkage split). This pass made no source changes; see below.

## What This Pass Did Not Find

This audit specifically looked for "small, cheap, clearly-in-spirit" fixes (a stale comment, a trivially
wireable connection) per the task's scope discipline, and did not find one. Every deferred/incomplete
piece traced back to either a genuine open design question or an intentional, already-documented
architectural boundary — a reflection of how consistently every `src/pcg/*` file already states its own
scope limits up front (see e.g. `PcgSurfaceSampler.h`'s "SCOPE BOUNDARY" comment, `PcgCellGenerator.h`'s
"Two-namespace CellCoord situation," `PcgVolumeInspector.h`'s "Scope decision" comment). No stale
comments, no orphaned/un-built source files (the main target's `file(GLOB_RECURSE ... "src/*.cpp")` in
`CMakeLists.txt` picks up every `src/pcg/*.cpp` and `src/renderer/debug/Pcg*.cpp` automatically — nothing
is silently excluded from the build), and no claim in any header comment that didn't hold up against the
actual code (spot-checked `AtmosClimatePass::GetEffectiveRelativeHumidity` specifically, since
`PcgClimateBiomeSelector.h` describes it as newly-added by that phase — confirmed real).

## Test Coverage (verified, not just claimed)

19 dedicated PCG CTest targets, ~6,350 lines of test code total (`tests/Pcg*.cpp`, `wc -l` verified),
each wired into `CMakeLists.txt` with an explicit, documented link-dependency rationale:
`PcgDataModelTests`, `PcgVolumeSamplerTests`, `PcgSplineSamplerTests`, `PcgSurfaceSamplerTests`,
`PcgTerrainSamplerTests`, `PcgSlopeHeightFilterTests`, `PcgSelfPruningFilterTests`, `PcgBiomeRulesTests`,
`PcgEcosystemExclusionTests`, `PcgClimateBiomeSelectorTests`, `PcgGraphEngineTests`,
`PcgBooleanSetOpsTests`, `PcgDensityTransformFilterTests`, `PcgNodePluginTests`, `PcgMeshSpawnerTests`,
`PcgGpuNodeExecutorTests`, `PcgVolumeActorTests`, `PcgCellGeneratorTests`, `PcgCellLoaderTests`. Plus 5
in-engine startup smoke tests surfaced through `--test-pipeline`'s report (`core::debug::DebugTestPipeline`,
Phase 9.2, already merged): PCG Phase 0.1 (Instance Registry), 0.2 (Instance Draw), 0.3 (Dynamic Lumen
registration), 4.2 (full Sampler→Filter→Spawner→Render pipeline), 6.3 (Runtime Generator Hook, which also
covers 6.4/6.5's caching and determinism-validation extensions to the same test).

---

## Sources

- [Procedural Content Generation Framework Node Reference in Unreal Engine — Unreal Engine 5.8 Documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-framework-node-reference-in-unreal-engine)
- [Procedural Content Generation Overview — Unreal Engine 5.8 Documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-overview)
