# Rapport d'Optimisation des Performances MegaLights - Unreal Engine 5.8

Ce rapport présente les profils de qualité et les paramètres d'optimisation appliqués aux fichiers `Config/DefaultEngine.ini` et `Config/DefaultScalability.ini` pour assurer des performances maximales et une qualité visuelle irréprochable du système **MegaLights**.

---

## 1. Profils de Qualité (Scalability Tiers)

Les profils de qualité pour MegaLights ont été définis dans le groupe de scalabilité `ShadowQuality` (niveaux Low à Cine) pour permettre au moteur de s'adapter automatiquement aux différentes configurations matérielles.

| Profil de Qualité | Variables de Configuration Principales | Rôle / Description |
| :--- | :--- | :--- |
| **Low (@0)** | `r.MegaLights=0`<br>`r.MegaLights.Allow=0`<br>`r.MegaLights.ShadowSamples=0` | MegaLights est entièrement désactivé sur les configurations d'entrée de gamme pour économiser le budget GPU. |
| **Medium (@1)** | `r.MegaLights.Allow=1`<br>`r.MegaLights.ShadowSamples=1`<br>`r.MegaLights.MaxLights=128`<br>`r.MegaLights.DownsampleMode=2` | Rendu MegaLights à demi-résolution spatiale avec budget de rayons de visibilité réduit. |
| **High (@2)** | `r.MegaLights.Allow=1`<br>`r.MegaLights.ShadowSamples=2`<br>`r.MegaLights.MaxLights=256`<br>`r.MegaLights.DownsampleMode=1` | Profil standard équilibré pour cartes graphiques grand public. |
| **Epic (@3)** | `r.MegaLights.Allow=1`<br>`r.MegaLights.ShadowSamples=4`<br>`r.MegaLights.MaxLights=512`<br>`r.MegaLights.DownsampleMode=0` | Haute fidélité visuelle, rendu pleine résolution avec une pénombre douce et précise. |
| **Cinematic (@4)** | `r.MegaLights.Allow=1`<br>`r.MegaLights.ShadowSamples=8`<br>`r.MegaLights.MaxLights=1024`<br>`r.MegaLights.DownsampleMode=0` | Qualité maximale de production et de rendu cinématique. |

---

## 2. Configuration du Débruiteur Spatio-Temporel (Denoiser)

Comme MegaLights utilise un échantillonnage stochastique par tracé de rayons, un filtrage spatio-temporel performant est indispensable. Les variables ont été configurées pour lisser le bruit tout en éliminant les traînées visuelles (**ghosting**) sur les objets rapides.

*   **`r.MegaLights.Temporal=1`** : Active l'accumulation temporelle des échantillons lumineux à travers les trames passées.
*   **`r.MegaLights.Temporal.MaxFramesAccumulated=8`** : Réduit l'historique d'accumulation de 32/64 trames (par défaut) à **8 trames**. Cela accélère la réactivité du débruiteur et élimine presque totalement le *ghosting* sur les mouvements rapides, tout en maintenant une stabilité temporelle satisfaisante.
*   **`r.MegaLights.Temporal.MinFramesAccumulatedForHighConfidence=4`** : Permet au système d'accorder rapidement une grande confiance aux nouvelles données géométriques.
*   **`r.MegaLights.Spatial=1`** : Active le filtre spatial dans l'espace écran pour nettoyer le bruit haute fréquence résiduel au sein d'une même trame.
*   **`r.MegaLights.Spatial.KernelRadius=4.0`** : Rayon du noyau de filtrage (4 pixels) pour étaler et adoucir la pénombre sans flouter les détails fins des ombres d'intersection.
*   **`r.MegaLights.Spatial.NumSamples=8`** : Échantillons spatiaux de filtrage pour une reconstruction de pénombre lisse et stable.

---

## 3. Culling Agressif de Lumière (Optimisation BVH)

Afin d'éviter que le GPU ne passe du temps à calculer la visibilité pour des lumières invisibles ou à contribution négligeable, des règles de culling agressives ont été instaurées :

*   **`r.ManyLights.Sampling.MinWeight=0.04`** & **`r.MegaLights.MinWeight=0.04`** : Filtre et exclut de l'échantillonnage d'importance stochastique toute lumière dont le poids de contribution énergétique à l'écran est inférieur à **4%** (seuil par défaut à 1%). Les lumières trop éloignées ou trop faibles sont ainsi immédiatement ciblées et ignorées sans aucun impact visuel perceptible.
*   **`r.MegaLights.MinWeightSum=0.05`** : Seuil de culling pour la somme cumulée des lumières environnantes.
*   **`r.RayTracing.Culling=1`** : Active le culling de scène pour le pipeline de Ray Tracing.
*   **`r.RayTracing.Culling.Radius=8000.0`** : Limite le culling à **80 mètres** de distance autour de la caméra pour exclure les micro-géométries du BVH.
*   **`r.RayTracing.Culling.Angle=1.0`** : Exclut du BVH de traçage tous les objets dont la taille angulaire projetée à l'écran est inférieure à **1 degré**, allégeant considérablement le coût des intersections de rayons pour MegaLights.
