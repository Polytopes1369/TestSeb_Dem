## Contexte du projet
Je développe une démoscene moderne Windows sans limitation stricte de taille mémoire (pas de contrainte 64kb), mais avec un souci d'optimisation du poids de l'exécutable (pas de frameworks lourds, aucun data dans mon .exe). L'objectif est la performance brute, la qualité visuelle et une architecture de rendu moderne.

## Description des besoins de la demoscene:
Tout est 100% procedural GPU driven.
* Arbres (generer par du code style speedtree)
* Terrain procedural, un peu de mer, quelques plages, montagnes, plaines, rivières, ruisseau, chutes d'eau, chemins, pierres, arbustres, et autres assets visuels. (generation style houdini et PCG).
* Generation de son/musique procedural (moteur de son 3D + style FL studio).
* Generation de textures procedurales (style substance)
* Generation de geometrie procedurale (style houdini aussi)
* Generation de FX proceduraux (style houdini)

## Cadre Technique
* **Langage & Outils :** C++23, CMake (cibles Windows x64).
* **Fenêtrage & Surface :** GLFW (liaison statique pour l'autonomie de l'exécutable).
* **API Graphique :** Vulkan 1.3+. Utilisation obligatoire du Dynamic Rendering (VK_KHR_dynamic_rendering).
* **Extensions Matérielles Requises :** Ray Tracing (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`) et Mesh Shaders (`VK_EXT_mesh_shader`).
* **Bindless pipeline :** On génère les meshes procéduralement par compute shader (primitive), on rasterize (ou ray-trace), et on affiche le tout sur la carte graphique avec un Descriptor Array massif (Bindless).
* **Gestion Mémoire :** Vulkan Memory Allocator (VMA). Architecture orientée Bindless et GPU-driven rendering.
* **Gestion des Erreurs :** Validation Layers activées en Debug. Vérification systématique de chaque VkResult avec crash explicite ou levée d'exception en cas d'échec.

## Règles d'interaction et de code
0. **Commentaires :** Tu dois bien commenter les parties complexes (logique Vulkan ardue, synchronisations, mathématiques 3D). Tous les commentaires au sein du code doivent être rédigés en **anglais**.
1. **Zéro approximation :** Pas de code d'exemple simplifié ou tronqué avec des commentaires du type "// Initialiser ici". Fournis des implémentations complètes, robustes et fonctionnelles.
2. **Focus Synchronisation :** La synchronisation Vulkan (VkBarrier2, VkDependencyInfo) est critique. Explicite toujours les structures de barrières de mémoire, de buffers et de transition de layout d'image sans raccourcis.
3. **Style de code :** C++ moderne (C++23), pas de pointeurs bruts pour la gestion de la durée de vie des ressources (RAII obligatoire), structures de données claires et alignées pour les Uniform/Storage Buffers (alignas/std430 compatible).
4. **Nouveaux fichiers :** Chaque fois que tu crées un fichier, donne son nom, son arborescence précise sous forme de bloc de texte clair, puis le code complet. Crée des dossiers structurés si nécessaire.
5. **Modification d'une fonction :** Donne explicitement le code "Avant correction" et "Après correction".
6. **Journalisation & Rigueur :** Utilise notre système de log unifié pour rapporter chaque étape critique (initialisation, allocations VMA, chargements, pipelines).
7. **Modifications Incrémentales :** Fais des modifications étape par étape et assure la logique de compilation à chaque fichier traité.