# Outil d'Automatisation de Conversion MegaLights pour Unreal Engine 5.8

Ce script Python est conçu pour automatiser la conversion et l'optimisation des sources lumineuses locales (Point, Spot et Rect Lights) vers le nouveau pipeline de rendu stochastique **MegaLights** sous Unreal Engine 5.8+.

---

## 1. Description du Script

Le script `MegaLightsConverter.py` effectue les opérations suivantes en une seule passe sur la scène active (ou un dossier World Outliner spécifié) :

1. **Identification & Filtrage** :
   * Détecte les composants de type `unreal.PointLightComponent`, `unreal.SpotLightComponent` et `unreal.RectLightComponent`.
   * Filtre uniquement les lumières définies en **Movable** (Mobiles) ou **Stationary** (Stationnaires) pour éviter de modifier la lumière statique ou cuite.
   * Ignore les sources globales telles que la `DirectionalLightComponent` ou la `SkyLightComponent`.
   * Permet de cibler uniquement un sous-dossier de l'Outliner via l'argument `folder_path`.

2. **Activation de MegaLights** :
   * Active la propriété `allow_mega_lights` (boîtier "Allow MegaLights" de l'éditeur).
   * Configure `mega_lights_shadow_method` sur `unreal.MegaLightsShadowMethod.RAY_TRACING` pour utiliser le traçage de rayons stochastique d'importance, qui offre le meilleur rapport performance/qualité sans surcharger le processeur avec des Virtual Shadow Maps.

3. **Désactivation des Méthodes d'Ombrage Obsolètes** :
   * Désactive les ombres par champ de distance (`use_ray_traced_distance_field_shadows = False`).
   * Désactive les ombres volumétriques obsolètes (`cast_volumetric_shadow = False`).
   * Désactive les ombres translucides traditionnelles (`cast_translucent_shadows = False`).
   * Cela évite les doublons de calculs d'ombres et allège les passes d'atténuation.

4. **Optimisation Mathématique de l'Atténuation** :
   * Réduit dynamiquement le rayon d'atténuation (`attenuation_radius`) en fonction de l'intensité de chaque lumière pour éviter que les volumes d'influence ne se chevauchent de manière excessive.
   * Les calculs adaptent le rayon selon la racine carrée de l'intensité de la lumière, avec des limites de clamping (`max_radius_clamp` et `min_radius_limit`) personnalisables, maximisant la performance du culling spatial (BVH GPU).

---

## 2. Utilisation dans l'Éditeur Unreal Engine

### Exécution depuis la Console Python de l'Éditeur
Pour exécuter le script dans Unreal Editor :
1. Ouvrez l'éditeur et affichez la console Python (**Window > Developer Tools > Python Console**).
2. Exécutez le script en copiant-collant son contenu ou en le chargeant depuis le disque :
   ```python
   import sys
   sys.path.append("D:/DemoScene/DemoScene_Vulkan2026_BaseArchi/DemoScene_2026/tools")
   import MegaLightsConverter
   
   # Convertit toutes les lumières de la scène avec les valeurs par défaut
   MegaLightsConverter.convert_local_lights_to_megalights()
   ```

### Paramètres de Personnalisation
La fonction prend en charge plusieurs paramètres :
```python
MegaLightsConverter.convert_local_lights_to_megalights(
    folder_path="/Lights/Local", # Restreint la recherche au dossier "/Lights/Local"
    max_radius_clamp=1000.0,     # Limite le rayon d'atténuation maximal à 10m (1000 cm)
    min_radius_limit=200.0       # Garantit que le rayon ne descend pas sous 2m (200 cm)
)
```

---

## 3. Exemple de Rapport de Sortie (Log Console)

Une fois le traitement terminé, le script génère un rapport formaté récapitulant les optimisations appliquées :

```text
LogPython: ==========================================================================================
LogPython:                     MEGALIGHTS CONVERSION & OPTIMIZATION REPORT                           
LogPython: ==========================================================================================
LogPython: Scan Scope: Entire Level
LogPython: Total Local Lights Converted to MegaLights: 5
LogPython: Total Attenuation Radius Reduction: 1250.00 Unreal Units
LogPython: ------------------------------------------------------------------------------------------
LogPython: Actor Name                    | Light Type           | Mobility   | Orig Rad   | Opt Rad    | Reduction 
LogPython: ------------------------------------------------------------------------------------------
LogPython: PointLight_Corridor_1         | PointLightComponent  | Movable    | 1000.0     | 565.7      | 434.3     
LogPython: PointLight_Corridor_2         | PointLightComponent  | Movable    | 1000.0     | 565.7      | 434.3     
LogPython: SpotLight_Desk                | SpotLightComponent   | Movable    | 800.0      | 800.0      | 0.0       
LogPython: RectLight_WallSign            | RectLightComponent   | Stationary | 1500.0     | 1200.0     | 300.0     
LogPython: PointLight_AmbientFill        | PointLightComponent  | Movable    | 2000.0     | 1200.0     | 800.0     
LogPython: ==========================================================================================
```
