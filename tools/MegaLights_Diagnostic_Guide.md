# Guide Diagnostic & Journal d'Erreurs Type MegaLights - UE 5.8

Ce document sert de journal d'erreurs type (template diagnostic) pour identifier, diagnostiquer et corriger rapidement les deux types d'artefacts visuels les plus courants sous le pipeline MegaLights : **les fuites de lumière (light leaking)** et **le sous-échantillonnage (under-sampling)**.

---

## Fiche d'Erreur 1 : Fuite de Lumière (Light Leaking)

### 1. Description du Phénomène
La lumière traverse des murs, des plafonds ou des géométries minces pourtant censées être opaques. Des halos lumineux ou des taches de lumière apparaissent dans des zones entièrement closes (ex: intérieurs fermés).

### 2. Symptômes Visuels
*   Lumières d'ambiance extérieures ou sources situées derrière une cloison visibles sur les surfaces intérieures.
*   Bords des ombres nettes décalés par rapport aux points de contact géométriques.

### 3. Causes Possibles (Root Causes)
*   **Géométrie trop fine (Thin Walls)** : L'arbre d'accélération BVH de Ray Tracing matériel simplifie ou ignore les surfaces dont l'épaisseur est trop faible lors de la construction des niveaux de détail.
*   **Backface Culling lors du Ray Tracing** : Par défaut, pour optimiser le traçage, les rayons ignorent les faces arrière (backfaces) de la géométrie, permettant à la lumière de s'infiltrer à travers les jonctions non scellées.
*   **Ray Tracing Normal Bias inadapté** : Un biais trop élevé pousse l'origine des rayons trop loin de la surface, sautant la géométrie de collision.

### 4. Code & Messages d'Alerte type (Log Output)
```text
LogRenderer: Warning: MegaLights - BVH intersection skipped due to normal bias mismatch on Actor 'StaticMeshActor_Wall2'.
LogRayTracing: Warning: Geometry 'SM_Wall_Thin' thickness is below hardware ray tracing threshold. Shadow leaks may occur.
```

### 5. Solutions & Variables de Correction (Action Plan)
*   **Activer le traçage double-face** pour forcer l'intersection des rayons sur les faces arrière :
    ```ini
    r.MegaLights.HardwareRayTracing.ForceTwoSided=1
    ```
*   **Ajuster le biais des normales** pour rapprocher l'origine des rayons de la surface physique sans introduire d'acné d'ombre :
    ```ini
    r.RayTracing.NormalBias=0.1
    ; ou pour MegaLights spécifiquement :
    r.MegaLights.NormalBias=0.1
    ```
*   **Épaissir la géométrie** : Augmenter l'épaisseur des cloisons à un minimum de 10 cm dans la modélisation 3D pour assurer leur inclusion dans le BVH matériel.

---

## Fiche d'Erreur 2 : Sous-Échantillonnage (Under-Sampling)

### 1. Description du Phénomène
Les ombres douces et les pénombres apparaissent granuleuses, bruitées, ou semblent "ramper" et scintiller lors des mouvements de caméra. Un effet de traînée visuelle (**ghosting**) est visible derrière les objets se déplaçant rapidement.

### 2. Symptômes Visuels
*   Bruit stochastique (grain type "sable") prononcé dans la pénombre des ombres portées.
*   Ombres qui se détachent ou se floutent de manière anormale en mouvement, laissant des traînées sombres (ghosting).
*   Scintillement (flickering) des spéculaires et des reflets indirects.

### 3. Causes Possibles (Root Causes)
*   **Budget de rayons insuffisant** : Trop peu de rayons de visibilité tracés par pixel (`r.MegaLights.ShadowSamples` trop bas).
*   **Accumulation temporelle excessive** : Le débruiteur mélange trop d'anciennes trames (MaxFramesAccumulated trop élevé), créant des traînées sur les objets rapides.
*   **Culling de poids trop agressif** : Les lumières faibles sont culled trop violemment, causant des apparitions/disparitions abruptes de bruit.

### 4. Code & Messages d'Alerte type (Log Output)
```text
LogRenderer: Warning: MegaLights - Light grid cell buffer overflow! Number of active overlapping lights exceeds sample allocation budget.
LogTemporalAA: Performance Warning: MegaLights stochastic sampling noise exceeds temporal denoiser convergence limit. Increase sample count.
```

### 5. Solutions & Variables de Correction (Action Plan)
*   **Augmenter le budget d'échantillonnage** (Shadow Samples) pour réduire le bruit natif à la source :
    ```ini
    r.MegaLights.ShadowSamples=4
    ; ou via la variable équivalente :
    r.MegaLights.NumSamplesPerPixel=4
    ```
*   **Réduire l'accumulation temporelle** pour éliminer le ghosting (au prix d'un bruit légèrement plus visible en mouvement, compensé par le filtrage spatial) :
    ```ini
    r.MegaLights.Temporal.MaxFramesAccumulated=8
    ```
*   **Ajuster le rayon du filtre spatial** pour étaler et lisser le bruit sur une plus grande zone d'écran :
    ```ini
    r.MegaLights.Spatial.KernelRadius=6.0
    r.MegaLights.Spatial.NumSamples=12
    ```
*   **Diminuer le seuil de culling de poids** pour stabiliser les lumières à faible contribution :
    ```ini
    r.ManyLights.Sampling.MinWeight=0.01
    ```
