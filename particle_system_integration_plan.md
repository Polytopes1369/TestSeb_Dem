# Plan d'Intégration d'un Système de Particules GPU (Niagara-style)

Ce document établit l'architecture technique, l'intégration aux pipelines existants (Nanite & Lumen) et le plan de développement d'un système de particules entièrement simulé et rendu sur le GPU dans un moteur C++23 / Vulkan 1.3+.

---

## 1. Architecture Générale et Synergie des Pipelines

Pour préserver les garanties anti-stuttering du moteur (soumission d'un unique command buffer sans synchronisations bloquantes CPU/GPU), le système de particules fonctionne en mode **100% GPU-Driven** :
1. **Simulation (Compute)** : Un compute shader met à jour l'état des particules, gère la naissance (spawning) et la mort via des listes d'index atomiques (`DeadList` / `AliveList`), applique les forces et résout les collisions physiques avec le terrain/objets en échantillonnant le pipeline **Global SDF** de l'engin.
2. **Tri (Compute)** : Tri de type *Bitonic Sort* ou *Radix Sort* sur le GPU pour trier les particules transparentes de l'arrière vers l'avant par rapport à l'axe de vue de la caméra afin de garantir un mélange alpha correct.
3. **Rendu (Graphics)** : Rendu par billboard dynamique ou mesh instancié sans Vertex Buffer traditionnel. Les coordonnées des quads sont générées à la volée dans le Vertex Shader via `gl_VertexIndex` en lisant le SSBO des particules triées.
4. **Intégration Lumen & VSM** : Les fragments de particules échantillonnent la grille **World Probe Grid** pour l'indirect diffuse, calculent le direct lighting via les sources lumineuses globales, reçoivent les ombres directionnelles de la cascade de **Virtual Shadow Maps (VSM)** et écrivent une distorsion de chaleur dans la texture de réfraction (`g_RefractionOffset`) lue par le pass de post-process.

---

## 2. Découpage en Sous-tâches & Prompts Associés

---

### Sous-tâche 1 : Architecture CPU/GPU & Buffers de Particules

#### Objectifs
1. Créer la structure `struct Particle` côté GPU contenant la position, la vitesse, la couleur, la taille, la rotation, la vie actuelle, la vie max et le random seed.
2. Mettre en place les buffers SSBO double-buffered pour les particules (`m_ParticleBuffer[2]`), l'indirect draw arguments buffer (`m_IndirectDrawBuffer`), le compteur de particules actives (`m_CounterBuffer`) et la dead-list.
3. Développer le squelette C++ de la classe `ParticleSystemPass` gérant l'allocation mémoire via **VMA** et les layouts de descripteurs.

#### Fichiers concernés
- **NEW** : `src/renderer/passes/ParticleSystemPass.h` / `.cpp`
- **NEW** : `src/shaders/src/Renderer/ParticleCommon.glsl`
- **MODIFY** : `src/renderer/ClusterRenderPipeline.h` / `.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 1
```text
Rôle : Expert C++23 / Vulkan 1.3 Graphics Architect.
Tâche : Crée les structures et l'infrastructure de buffers pour notre système de particules GPU-driven.

1. Crée un fichier GLSL partagé `ParticleCommon.glsl` déclarant la structure de particule compatible std430 (64 octets, alignement strict 16 octets) :
   struct Particle {
       vec3 position; float life;
       vec3 velocity; float maxLife;
       vec4 color;
       vec2 size; float rotation; uint randomSeed;
   };
   Et les structures de buffers :
   layout(std430, set = 0, binding = 0) buffer ParticleBuffer { Particle particles[]; };
   layout(std430, set = 0, binding = 1) buffer DeadListBuffer { uint deadIndices[]; };
   layout(std430, set = 0, binding = 2) buffer AliveListBuffer { uint aliveIndices[]; };
   layout(std430, set = 0, binding = 3) buffer CounterBuffer { uint deadCount; uint aliveCount; uint spawnQueue; uint _pad0; };

2. Écris la classe C++ `ParticleSystemPass` (fichiers `ParticleSystemPass.h` et `.cpp`) encapsulant :
   - L'initialisation (Init) prenant le VkDevice, le VmaAllocator et d'autres ressources communes.
   - La gestion mémoire VMA de deux Particle Buffers GPU_ONLY (double-buffering), de la dead-list, de l'alive-list et du counter buffer.
   - L'allocation et l'écriture d'un buffer d'arguments indirects `VkDrawIndirectCommand`.
   - La création du Descriptor Pool et du Descriptor Set Layout reliant les ressources.
   - Les méthodes `Shutdown()` libérant proprement les ressources Vulkan/VMA via RAII.

3. Intègre le pass dans `ClusterRenderPipeline` (Init et Shutdown). Le système doit être compilé de manière conditionnelle en fonction des types de build (séparation stricte Debug vs Release). Exclus toutes les structures et chaînes de debug en Release.

Fournis les codes C++23 et GLSL complets et robustes, sans aucun raccourci ni commentaire "TODO". Active explicitement les vérifications d'erreurs Vulkan.
```

---

### Sous-tâche 2 : Simulation Compute Pipeline (Spawning, Update, Wind & SDF Collisions)

#### Objectifs
1. Écrire le compute shader `ParticleSimulation.comp` exécutant la simulation physique par thread de travail.
2. Implémenter le module de spawn : lit la file d'attente de spawn, retire des index de la dead-list, initialise l'état des nouvelles particules, et les injecte dans l'alive-list.
3. Appliquer la gravité, l'amortissement aérodynamique (drag) et les forces de vent du système **Atmos** (`SampleWindVelocity`).
4. Résoudre les collisions rigides avec la géométrie du monde en échantillonnant le clipmap **Global SDF** de l'engin : calculer la normale locale par différence centrale si $SDF < 0$ pour repousser la particule et appliquer un coefficient de rebond et de friction.

#### Fichiers concernés
- **NEW** : `src/shaders/src/Renderer/ParticleSimulation.comp`
- **MODIFY** : `src/renderer/passes/ParticleSystemPass.h` / `.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 2
```text
Rôle : Expert C++23 / Vulkan 1.3 Graphics Architect.
Tâche : Développe le pipeline de simulation compute pour le système de particules GPU-driven.

1. Écris le compute shader `ParticleSimulation.comp` qui prend en entrée l'état actuel et écrit l'état futur (double buffering) :
   - local_size_x = 64.
   - Gère le spawn en lisant `spawnQueue`. Chaque thread prenant en charge une nouvelle particule consomme un index dans `deadIndices`, initialise sa position à l'émetteur, réinitialise sa vie/maxLife et calcule un seed aléatoire.
   - Met à jour les particules existantes : `position += velocity * dt`.
   - Échantillonne le vent de la scène en incluant `AtmosNoiseCommon.glsl` ou calcule localement un bruit de Curl 3D pour perturber la vitesse de la particule.
   - Intègre les collisions physiques par rapport au terrain avec le clipmap Global SDF :
     * Appelle une fonction `float SampleGlobalSDF(vec3 worldPos)` échantillonnant la texture 3D de `GlobalSDFPass`.
     * Si la distance retournée est négative (pénétration), calcule la normale par différence centrale :
       n = normalize(vec3(SampleGlobalSDF(worldPos + vec3(e,0,0)) - SampleGlobalSDF(worldPos - vec3(e,0,0)), ...))
       Repousse la particule hors du volume et réfléchis la vitesse relative par rapport à la normale (`reflect(v, n)`) modulée par l'élasticité et la friction.
     * Si la vie de la particule expire, ajoute son index à la dead-list en décrémentant de façon atomique le compteur actif.

2. Côté C++ dans `ParticleSystemPass`, initialise le pipeline compute (PSO) pour `ParticleSimulation.comp` et implémente la méthode `RecordSimulate(VkCommandBuffer cmd, float dt, const vec3& emitterPosition)`. Ajoute les barrières mémoire nécessaires pour synchroniser l'écriture du SSBO de particules par le compute shader avant d'autres lectures graphiques ou de tri.

Fournis les codes C++23 et GLSL complets et prêts à être compilés, respectant les normes de codage rigoureuses du moteur.
```

---

### Sous-tâche 3 : GPU Particle Sorting Pipeline

#### Objectifs
1. Écrire un trieur GPU efficace pour trier les particules translucides de l'arrière vers l'avant (back-to-front).
2. Implémenter un algorithme de tri bitonique ou de tri par base (Radix Sort) dans `ParticleSort.comp` qui prend en entrée la distance caméra de chaque particule et produit un tableau d'index triés.
3. Calculer la clé de tri de chaque particule vivante en projetant sa position sur le vecteur forward de la caméra : $Key = dot(ParticlePos - CameraPos, CameraForward)$.

#### Fichiers concernés
- **NEW** : `src/shaders/src/Renderer/ParticleSort.comp`
- **MODIFY** : `src/renderer/passes/ParticleSystemPass.h` / `.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 3
```text
Rôle : Expert C++23 / Vulkan 1.3 / GPU-Driven Architecture.
Tâche : Développe le pipeline de tri GPU (Radix ou Bitonic) pour trier les particules transparentes du fond vers le premier plan.

1. Écris le compute shader `ParticleSort.comp` :
   - Étape A (InitKeys) : Calcule la clé de tri pour chaque particule active (`aliveCount`). La clé est la distance projetée sur l'axe optique de la caméra : `float depth = dot(particle.position - cameraPositionWorld, cameraForwardVector)`. Stocke une paire structurée `(uint index, float key)` dans un buffer temporaire.
   - Étape B (SortPasses) : Implémente un tri GPU parallèle (ex. un Bitonic Sort par passes successives) triant par ordre décroissant de profondeur (les plus éloignées en premier, pour l'échantillonnage de transparence "over").

2. Côté CPU dans `ParticleSystemPass` :
   - Alloue les buffers de clés/index temporaires requis pour le tri.
   - Crée le pipeline de tri GPU.
   - Implémente la méthode `RecordSort(VkCommandBuffer cmd, const vec3& cameraPos, const vec3& cameraForward)` qui orchestre les dispatches successifs nécessaires selon le nombre actuel de particules à trier (qui peut être dynamique, arrondi à la puissance de 2 supérieure pour le Bitonic Sort).
   - Insère les barrières Vulkan adéquates (`VkMemoryBarrier2` avec `COMPUTE_SHADER_BIT` en source et destination) entre chaque passe de tri.

Rédige le code 100% complet. Aucun pseudo-code ni fonction tronquée n'est accepté.
```

---

### Sous-tâche 4 : Graphics Pipeline & Rendu de Billboards

#### Objectifs
1. Créer le pipeline graphique de rendu dans `ParticleSystemPass` en utilisant le rendu dynamique (`VK_KHR_dynamic_rendering`).
2. Implémenter le Vertex Shader `ParticleRender.vert` : ne pas lier de Vertex Buffer classique. Utiliser le `gl_VertexIndex` (0 à 5) pour générer les coordonnées de quad à la volée. Lire le SSBO trié des particules pour positionner chaque quad dans le monde.
3. Implémenter le Fragment Shader `ParticleRender.frag` : échantillonner la texture de la particule et gérer la transparence. Gérer le fondu de bordure (soft particles) en comparant le Z du fragment actuel avec le Z de la scène opaque stocké dans le depth buffer global.

#### Fichiers concernés
- **NEW** : `src/shaders/src/Renderer/ParticleRender.vert` / `.frag`
- **MODIFY** : `src/renderer/passes/ParticleSystemPass.h` / `.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 4
```text
Rôle : Expert C++23 / Vulkan 1.3 Graphics Architect.
Tâche : Développe le pipeline de rendu graphique dynamique pour afficher les particules sous forme de quads orientés (billboards).

1. Écris le Vertex Shader `ParticleRender.vert` :
   - Génère les coordonnées du quad dynamiquement sans Vertex Buffer lié :
     * `gl_VertexIndex` génère 6 sommets (2 triangles) de coordonnées UV : (0,0), (1,0), (0,1), (0,1), (1,0), (1,1).
     * Récupère l'index de la particule triée via le buffer produit à la sous-tâche 3.
     * Récupère la position monde, la taille et la rotation de la particule.
     * Calcule la position dans l'espace caméra en appliquant l'orientation face caméra (billboarding cylindrique ou sphérique) :
       vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
       vec3 up = vec3(view[0][1], view[1][1], view[2][1]);
       vec3 billboardPos = particle.position + (uv.x - 0.5) * particle.size.x * right + (uv.y - 0.5) * particle.size.y * up;
       gl_Position = proj * view * vec4(billboardPos, 1.0);

2. Écris le Fragment Shader `ParticleRender.frag` :
   - Échantillonne une texture de texture-atlas bindless pour définir la forme (ex. fumée, étincelle).
   - Calcule l'atténuation "Soft Particles" en lisant le depth buffer global (reversed-Z) :
     * Décompresse la profondeur de l'arrière-plan `sceneZ` et calcule la distance caméra correspondante.
     * Calcule la distance caméra du fragment de la particule `particleZ = gl_FragCoord.z`.
     * Applique un fondu linéaire de l'opacité selon la différence `sceneZ - particleZ`.
   - Effectue le mélange final avec fixed-function alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) configuré dans la PSO C++.

3. Dans `ParticleSystemPass`, crée la PSO graphique (`VkPipeline`) en utilisant le rendu dynamique (`VK_KHR_dynamic_rendering`). configure l'état de blending et le depth-test (VK_COMPARE_OP_GREATER, Reversed-Z) avec écriture de profondeur désactivée (`depthWriteEnable = VK_FALSE`). Implémente la méthode `RecordDraw(VkCommandBuffer cmd, VkImageView targetColorView, VkImageView depthView)`.

Renvoie les fichiers de code complets et bien formatés.
```

---

### Sous-tâche 5 : Compatibilité Lumen, VSM & Réfraction

#### Objectifs
1. Intégrer l'éclairage physique aux particules transparentes dans le shader `ParticleRender.frag` :
   - Échantillonner la direction de la lumière du soleil et calculer la direct radiance.
   - Échantillonner l'atlas d'ombres **Virtual Shadow Maps (VSMs)** pour projeter des ombres sur les particules.
   - Échantillonner la grille **World Probe Grid** de Lumen (`SampleWorldProbeGrid`) pour appliquer l'éclairage indirect diffus sur le volume de particules.
2. Permettre aux particules de chaleur ou de distorsion d'écrire des vecteurs de déformation dans la texture RG16F `g_RefractionOffset` du pass de post-process afin de simuler des effets thermiques (heat shimmer).

#### Fichiers concernés
- **MODIFY** : `src/shaders/src/Renderer/ParticleRender.frag`
- **MODIFY** : `src/renderer/passes/ParticleSystemPass.h` / `.cpp`
- **MODIFY** : `src/renderer/ClusterRenderPipeline.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 5
```text
Rôle : Expert C++23 / Vulkan 1.3 / Lumen & VSM Developer.
Tâche : Ajoute le support des ombres VSM, de l'indirect diffuse Lumen et de la réfraction thermique à notre système de particules.

1. Modifie le Fragment Shader `ParticleRender.frag` pour implémenter l'éclairage complet des particules :
   - Échantillonne le soleil ombré par VSM en incluant les helpers `shadow_page_table.glsl`, `shadow_atlas_sampling.glsl` et `shadow_sun_sampling.glsl`.
   - Échantillonne l'éclairage indirect de Lumen en intégrant `world_probe_sampling.glsl` :
     vec3 indirectDiffuse = SampleWorldProbeGrid(worldPosition, normal);
     Lit la structure `WorldProbeGridParamsBuffer` pour localiser le volume.
   - Calcule le lighting de diffusion volumétrique simplifié (Henyey-Greenstein phase function ou Lambertian modifié) combinant le soleil ombré et l'indirect diffuse.
   - Gère le heat distortion : si le matériau de la particule est marqué "refractive", calcule un offset ondulatoire temporel (basé sur le `globalTime` de `WPOGlobalsBuffer`) et écris-le dans le second render target de la passe (`layout(location = 1) out vec2 outRefractionOffset`), qui sera lu par `PostProcessComposite.comp`.

2. Côté C++ dans `ParticleSystemPass`, mets à jour les descripteurs graphiques pour lier l'atlas VSM, la table de pages VSM, la grille 3D World Probe et son buffer de paramètres associés.

3. Dans `ClusterRenderPipeline::Init` et `RecordFrame`, mets à jour l'appel à `m_ParticleSystem.Init` et `RecordDraw` pour passer ces ressources système additionnelles de la même manière que `TransparentForwardPass::Init` les consomme déjà.

Renvoie le code GLSL et C++ complet et mis à jour pour cette phase de rendu avancée.
```

---

### Sous-tâche 6 : Intégration Finale, Panel ImGui & Validation

#### Objectifs
1. Instancier et exécuter le système de particules dans le squelette de l'engin dans `ClusterRenderPipeline::RecordFrame` après la passe opaque Nanite et la passe `m_TransparentForward`, mais avant la passe de post-process.
2. Ajouter des sliders dans le panneau de contrôle ImGui de `src/main.cpp` (Debug-only) pour piloter en temps réel l'émetteur : spawn rate, vitesse de projection, élasticité du rebond, friction SDF, couleur et durée de vie.
3. Afficher les statistiques de performance (nombre de particules actives, temps GPU de simulation et de tri) dans le HUD `DebugTextOverlay`.
4. Effectuer des tests de validation Vulkan stricts en Debug pour garantir l'absence de fuites mémoires, de descripteurs invalides ou de conflits de synchronisation (barrières de transition).

#### Fichiers concernés
- **MODIFY** : `src/renderer/ClusterRenderPipeline.h` / `.cpp`
- **MODIFY** : `src/main.cpp`
- **MODIFY** : `src/renderer/debug/DebugTextOverlay.h` / `.cpp`

#### 📋 Prompt Claude Code - Sous-tâche 6
```text
Rôle : Expert C++23 / Vulkan 1.3 Engine Architect.
Tâche : Finalise l'intégration du système de particules dans la boucle principale et ajoute les outils de diagnostic/debug.

1. Dans `ClusterRenderPipeline::RecordFrame`, séquence l'exécution du système de particules :
   - Étape A : Appelle `m_ParticleSystem.RecordSimulate(...)` après l'upload des constantes de frame.
   - Étape B : Appelle `m_ParticleSystem.RecordSort(...)` juste après pour trier les particules de ce frame.
   - Étape C : Appelle `m_ParticleSystem.RecordDraw(...)` dans le bloc de rendu forward [13c], dessinant sur l'image colorée de `m_GIComposite` avec le depth buffer en lecture seule. Assure-toi que les barrières de transition pour l'image colorée et le depth buffer couvrent l'exécution de la passe.

2. Dans `src/main.cpp`, intègre dans l'onglet ImGui (Debug-only) des Sliders pour modifier dynamiquement les paramètres système (`ParticleEmitterSettings` struct comprenant : spawn rate, max particles, gravity vector, bounce coefficient, wind influence, color variance).

3. Dans `DebugTextOverlay.cpp`, modifie `BuildFrameText` pour inclure la ligne :
   "GPU Particles: <activeParticlesCount> / <maxParticlesCount> (Sim+Sort+Draw: <time> ms)"
   (En lisant les données mesurées ou le compteur GPU). Exclus complètement ce code et cette chaîne en mode Release pour préserver la règle "Zero String Overhead in Release" de la solution.

4. Analyse le cycle de vie complet du système de particules en Debug avec les couches de validation Vulkan actives. Vérifie les barrières pour t'assurer qu'aucun warning n'est émis.

Fournis les modifications complètes de code C++23 sous forme de blocs complets.
```

---

## 3. Plan de Vérification & Validation de la Feature

### Tests Automatisés
- Lancer le projet en Debug pour s'assurer de l'absence totale d'alertes dans les couches de validation Vulkan (Validation Layers).
- Valider la compilation complète du projet sous CMake en configurant `-DCMAKE_BUILD_TYPE=Release` pour s'assurer de l'exclusion physique totale des fonctionnalités d'ImGui, des profilers et des chaînes de log liées aux particules.

### Vérification Visuelle Manuelle
1. **Collision SDF** : Vérifier que les particules rebondissent correctement sur le sol et sur l'Icosphère tessellée en cours de rotation.
2. **Tri Alpha** : Positionner la caméra de manière à chevaucher plusieurs émetteurs de fumée colorée translucide et vérifier visuellement l'absence de clignotement ou d'inversion dans l'ordre d'affichage.
3. **Heat Shimmer** : Activer la déformation thermique sur l'émetteur de feu/chaleur et confirmer que la scène en arrière-plan ondule de manière fluide et cohérente à l'écran.
