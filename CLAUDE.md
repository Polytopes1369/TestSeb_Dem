# CLAUDE.md - Directives du Projet Demoscene Vulkan 2026

## Contexte du projet
Je développe une démoscene moderne Windows sans limitation stricte de taille mémoire (pas de contrainte 64kb), mais avec un souci d'optimisation du poids de l'exécutable (pas de frameworks lourds, aucun data dans mon .exe). L'objectif est la performance brute, la qualité visuelle et une architecture de rendu moderne.

## Description des besoins de la demoscene:
Tout est 100% procedural GPU driven.
* Arbres (générés par du code style speedtree).
* Terrain procédural, un peu de mer, quelques plages, montagnes, plaines, rivières, ruisseau, chutes d'eau, chemins, pierres, arbustes, et autres assets visuels (génération style houdini et PCG).
* Génération de son/musique procédurale (moteur de son 3D + style FL studio).
* Génération de textures procédurales (style substance).
* Génération de géométrie procédurale (style houdini aussi).
* Génération de FX procéduraux (style houdini).

## Cadre Technique
* **Langage & Outils :** C++23, CMake (cibles Windows x64).
* **Fenêtrage & Surface :** GLFW (liaison statique pour l'autonomie de l'exécutable).
* **API Graphique :** Vulkan 1.3+. Utilisation obligatoire du Dynamic Rendering (`VK_KHR_dynamic_rendering`).
* **Extensions Matérielles Requises :** Ray Tracing (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`) et Mesh Shaders (`VK_EXT_mesh_shader`).
* **Bindless pipeline :** On génère les meshes procéduralement par compute shader (primitive), on rasterize (ou ray-trace), et on affiche le tout sur la carte graphique avec un Descriptor Array massif (Bindless).
* **Gestion Mémoire :** Vulkan Memory Allocator (VMA). Architecture orientée Bindless et GPU-driven rendering.
* **Gestion des Erreurs :** Validation Layers activées en Debug. Vérification systématique de chaque `VkResult` avec crash explicite ou levée d'exception en cas d'échec.

## Règles d'interaction et de code
0. **Commentaires :** Tu dois bien commenter les parties complexes (logique Vulkan ardue, synchronisations, mathématiques 3D). Tous les commentaires au sein du code doivent être rédigés en **anglais** (`// English comments only`).
1. **Zéro approximation :** Pas de code d'exemple simplifié ou tronqué avec des commentaires du type `"// Initialiser ici"`. Fournis des implémentations complètes, robustes et fonctionnelles.
2. **Focus Synchronisation :** La synchronisation Vulkan (`VkBarrier2`, `VkDependencyInfo`) est critique. Explicite toujours les structures de barrières de mémoire, de buffers et de transition de layout d'image sans raccourcis.
3. **Style de code :** C++ moderne (C++23), pas de pointeurs bruts pour la gestion de la durée de vie des ressources (RAII obligatoire, `std::unique_ptr`, `std::shared_ptr`), structures de données claires et alignées pour les Uniform/Storage Buffers (`alignas(16)` ou compatible `layout(std430)`).
4. **Nouveaux fichiers :** Chaque fois que tu crées un fichier, donne son nom, son arborescence précise sous forme de bloc de texte clair, puis le code complet. Crée des dossiers structurés si nécessaire.
5. **Modification d'une fonction :** Donne explicitement le code "Avant correction" et "Après correction" pour chaque modification.
6. **Journalisation & Rigueur :** Utilise notre système de log unifié pour rapporter chaque étape critique (initialisation, allocations VMA, chargements, modifications de pipelines).
7. **Modifications Incrémentales :** Fais des modifications étape par étape et assure la logique de compilation à chaque fichier traité.
8. Outils de Débogage & Séparation Build (Debug/Release Strict) :
   * **Règle d'or d'exclusion binaire (Release) :** Tout le code lié aux tests, à la validation GPU, aux overlays statistiques, au système de routage d'inputs Numpad, aux modes de visualisation (Lumen/Nanite) et au logger unifié **ne doit pas être compilé** en mode Release.
   * **Isolation par fichiers & CMake :** Les fichiers sources dédiés au debug (ex: `DebugInputRouter.cpp`, `Logger.cpp`, `GpuDebugBuffer.cpp`) doivent être exclus de la compilation via des conditions de build CMake (`if(CMAKE_BUILD_TYPE STREQUAL "Debug")`) ou encapsulés par une garde de précompilation englobant l'intégralité du fichier (`#ifdef _DEBUG ... #endif`). En mode Release, le compilateur ne doit générer *aucun* code objet pour ces fonctionnalités.
   * **Zéro overhead de chaînes de caractères :** Aucune chaîne de texte verbeuse, aucun formatage de diagnostic, et aucun symbole lié aux outils de debug ne doit persister dans l'exécutable final de production afin de garantir un poids minimal et une optimisation maximale.
   
## IMPORTANT : Avant de commencer à analyser ou à modifier le code, tu dois impérativement respecter ce protocole strict :
1. Crée et bascule sur une nouvelle branche Git locale isolée (SANS la pousser sur GitHub/Remote) en utilisant un nom descriptif basé sur la tâche.
!!! Crée un répertoire de travail séparé. !!! 
2. Effectue tout ton travail de modification et de test uniquement sur cette branche.
3. Une fois la tâche entièrement résolue et validée, effectue un commit de tes changements sur cette branche locale.
4. Enfin, bascule à nouveau sur la branche d'origine (la branche courante actuelle) et fusionne (merge) ta branche locale temporaire dedans afin que le résultat final se retrouve directement dans mon espace de travail actif. Supprime ensuite la branche temporaire locale que tu as créée.

Confirme que tu as créé la branche locale avant de formuler ta première proposition technique.

---

## État Actuel du Projet (2026-07-18)

### Vue d'ensemble rapide
**DemoScene Vulkan 2026** est un moteur GPU-driven 100% procédural pour Windows/Vulkan 1.3, conçu pour créer une démoscene haute performance. Le projet contient **238 fichiers sources**, **50+ passes de rendu**, et **multiple roadmaps en cours**.

### Roadmaps Complétées ✓
- **Nanite-Parity** (9 phases) : Cluster LOD, hardware/software rasterization, tessellation WPO
- **Lumen-Parity** (6+ phases) : Global SDF, Surface Cache, World Probe Grid, Screen-space GI, Reflections
- **Atmos Weather** (5 phases) : Climate model, PBR sky, volumetric clouds, froxel fog, cloud shadows
- **Particle System** (Niagara-parity, 6 subtasks) : Multi-emitter, mesh/ribbon/sprite rendering, collision, data interfaces

### Roadmaps En Cours 🔄
- **PCG Framework** (10 phases, 70% done)
  - ✓ Phases 0-3 (Data model, sampler, filters)
  - ✓ Phases 5.1-5.2 (Graph engine CPU)
  - ⏳ Phases 4, 6+ (GPU spawner, GPU vertex gen)
  - ⏳ Phase 7-10 (Editor, material binding, mesh ops)

- **UE5.8 Render Parity** (G1-G10, 70% done)
  - ✓ G2 (MegaLights), G8 (Post-process)
  - 🔄 G1, G3-G4, G5-G7, G10 (Various render features)

### Statistiques Clés
| Métrique | Valeur |
|----------|--------|
| Fichiers sources | 238 (.cpp/.h/.hpp) |
| Shaders GLSL | ~40+ (.comp/.glsl → SPIR-V) |
| Passes de rendu | 50+ |
| Tests (`--test-pipeline`) | 10/10 ✓ |
| Build (clean) | ~45s |
| Commits | 200+ |
| Feature branches actives | 15+ |

### Architecture Majeure
```
Main Rendering Pipeline:
  HZBPass → ClusterCullingPass → ClusterOcclusionCullingPass
    ↓
  ClusterHardwareRasterPass / ClusterSoftwareRasterPass
    ↓
  ClusterResolvePass (Visibility Buffer)
    ↓
  [Parallel] MegaLightsPass, GlobalSDFPass, SurfaceCachePass
    ↓
  [GI] ScreenTracePass → GICompositePass → WorldProbeGridPass
    ↓
  [Effects] AtmosClimatePass, AtmosCloudsPass, ParticleSystemPass
    ↓
  [Forward] TransparentForwardPass, WaterForwardPass
    ↓
  [Post] PostProcessPass → BloomPass → TAATSRPass → Present
```

### Structure des Sources
```
src/
  ├── main.cpp (1914 lignes, entry point)
  ├── core/ (Camera, Config, Logging, LoadingManager, ECS)
  ├── renderer/ (ClusterRenderPipeline + 50+ passes)
  │   ├── vulkan/ (VMA, command buffers, synchronization)
  │   ├── passes/ (Culling, Rasterization, Lighting, GI, FX)
  │   ├── streaming/ (LOD, residency, virtual textures)
  │   └── debug/ (ImGui overlays, Debug-only)
  ├── pcg/ (~25 fichiers, graph evaluation engine)
  ├── shaders/ (~40+ .comp/.glsl → SPIR-V)
  ├── audio/ (Procedural sound generator)
  ├── world/ (Scene graph, world partition streaming)
  ├── geometry/ (Mesh structures, compression)
  └── io/ (Asset loading, cache, async decompression)
```

### Prochaines Étapes Critiques (Ordre priorité)

**Immédiat (cette semaine)**
1. **PCG Phase 4** - GPU Spawner System
   - Compute shader spawner
   - Instance data generation
   - Collision awareness
2. **UE5.8 Gap G1** - Material Parameter Updates
   - Paramètres de matériau dynamiques
   - Bindless material table expansion

**Court-terme (2 semaines)**
3. **PCG Phase 6** - GPU Vertex Generation
4. **UE5.8 Gaps G3-G4** - Decals + Substrate SSS (en parallèle)
5. **Feature Branch Backlog** - Cluster DAG, Decals, Skeletal Animation

**Moyen-terme (1 mois)**
6. **PCG Phase 7** - Editor Tools (Node canvas complet)
7. **UE5.8 Gaps G5-G10** - Closure complète
8. **Performance Optimization Pass** - GPU memory, cache coherency

### Points d'attention connus
- **DAG-build hang** → Mitigé via LoadingManager throttle (4 workers)
- **Shader stale deploy** → Vérifier timestamps shaders_gen vs shaders/
- **Incremental build corruption** → Fresh dir + full rebuild si device-lost
- **TAA flicker** → Mitigation via TAATSR alpha cap (attend Phase B temporal)
- **Cluster bounds rotation** → Never in persisted SSBO, only local copy

### Documentation Complète
Voir [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) pour l'analyse détaillée complète du projet.

---

## Directives de Codage (Détaillé)