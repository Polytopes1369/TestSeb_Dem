## Contexte du projet
Je développe une démoscene moderne Windows sans limitation stricte de taille mémoire (pas de contrainte 64kb), mais avec un souci d'optimisation du poids de l'exécutable (pas de frameworks lourds, aucun data dans mon .exe). 
L'objectif est la performance brute, la qualité visuelle et une architecture de rendu moderne.

## Cadre Technique
* Langage & Outils : C++23, CMake (cibles Windows x64).
* Fenêtrage & Surface : GLFW (liaison statique pour l'autonomie de l'exécutable).
* API Graphique : Vulkan 1.3+. Utilisation obligatoire du Dynamic Rendering (VK_KHR_dynamic_rendering).
* Extensions Matérielles Requises : Ray Tracing (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`) et Mesh Shaders (`VK_EXT_mesh_shader`).
* Blindless pipeline: On genere les mesh proceduralement par compute shader (primitive), on rasterize, et on affiche le mesh tout sur la carte graphique. 
* Gestion Mémoire : Vulkan Memory Allocator (VMA). 
Architecture orientée Bindless et GPU-driven rendering. Une structure Bindless avec un Descriptor Array massif. 
* Gestion des Erreurs : - Validation Layers activées en Debug. Vérification systématique de chaque VkResult avec crash explicite ou levée d'exception en cas d'échec.


## Règles d'interaction et de code
0. Commentaires: Tu dois bien commenter les chiaes pertinentes. Tous les commentaires doivent etre rédigés en anglais
1. Zéro approximation : Pas de code d'exemple simplifié ou tronqué avec des commentaires du type "// Initialiser ici". Fournis des implémentations complètes et fonctionnelles.
2. Focus Synchronisation : La synchronisation Vulkan (VkBarrier2, VkDependencyInfo) est critique. Explicite toujours les structures de barrières de mémoire et de transition de layout d'image sans raccourcis.
3. Style de code : C++ moderne, pas de pointeurs bruts pour la gestion des ressources, utilisation de structures de données claires et alignées pour les Uniform/Storage Buffers.
4. Nouveaux fichiers: Chaque fois que je dois creer un fichier, donne moi son nom et son arborescence + le code complet
5. Modification d'une fonction: donne le code avant correction et après correction. 