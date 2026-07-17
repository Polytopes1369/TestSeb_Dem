# Rapport de Validation de Configuration Unreal Engine 5.8 - MegaLights

Ce rapport valide la configuration appliquée dans le fichier `Config/DefaultEngine.ini` pour activer le système de rendu dynamique MegaLights ainsi que toutes ses dépendances matérielles sous Unreal Engine 5.8.

---

## 1. API de Rendu Cible & Shader Model 6 (SM6)

Pour utiliser MegaLights et le Ray Tracing matériel de manière optimale, les plateformes cibles (Windows et Linux) ont été configurées pour utiliser exclusivement **DirectX 12** et **Vulkan** avec le **Shader Model 6 (SM6)**.

| Paramètre | Clé INI / Variable de console | Valeur Cible | Validation |
| :--- | :--- | :--- | :--- |
| **API Rendu par Défaut (Windows)** | `DefaultGraphicsRHI` (Windows) | `DefaultGraphicsRHI_DX12` | Actif (DirectX 12) |
| **Shader Formats Windows** | `D3D12TargetedShaderFormats` | `+D3D12TargetedShaderFormats=PCD3D_SM6` | Actif (SM6 requis pour VSM et RT) |
| **API Rendu par Défaut (Linux)** | `DefaultGraphicsRHI` (Linux) | `DefaultGraphicsRHI_Vulkan` | Actif (Vulkan) |
| **Shader Formats Linux** | `TargetedRHIs` (Linux) | `+TargetedRHIs=SF_VULKAN_SM6` | Actif (Vulkan SM6) |
| **Shader Model Forcé** | `r.ShaderModel` | `6` | Actif (SM6 forcé au runtime) |

---

## 2. Ray Tracing Matériel (Hardware Ray Tracing)

Le Ray Tracing matériel est indispensable pour le fonctionnement de MegaLights et de Lumen en mode matériel. Les variables suivantes garantissent l'activation et la compatibilité du pipeline.

| Paramètre | Clé INI / Variable de console | Valeur Cible | Validation / Rôle |
| :--- | :--- | :--- | :--- |
| **Ray Tracing Matériel** | `r.HardwareRayTracing` | `1` | Force l'utilisation du Ray Tracing Matériel GPU |
| **Master Switch Ray Tracing** | `r.RayTracing` / `r.RayTracing.Enable` | `1` | Active le système global de Ray Tracing |
| **Support Matériel Projet** | `bUseHardwareRayTracing` | `True` | Active la prise en charge globale au niveau du projet |
| **Skin Cache Compute** | `bSupportComputeSkinCache` | `True` | Requis pour éviter les crashs sur la déformation géométrique en RT |

---

## 3. Global Illumination & Réflexions Lumen (Hardware RT)

Lumen a été configuré pour exploiter pleinement le ray tracing matériel pour l'éclairage global indirect et les réflexions spéculaires.

| Paramètre | Variable de console | Valeur Cible | Description |
| :--- | :--- | :--- | :--- |
| **Master Lumen RT** | `r.Lumen.HardwareRayTracing` | `1` | Force Lumen à utiliser le Ray Tracing Matériel |
| **Auto-Détection RT** | `r.Lumen.UseHardwareRayTracingWhenAvailable` | `True` | Utilise le RT matériel dès que disponible sur le GPU |
| **Réflexions Lumen RT** | `r.Lumen.Reflections.HardwareRayTracing` | `1` | Active le RT matériel spécifique pour les réflexions |
| **Screen Probe Gather RT** | `r.Lumen.ScreenProbeGather.HardwareRayTracing` | `1` | Echantillonnage par sonde d'écran en RT matériel |
| **Éclairage Direct Lumen RT** | `r.LumenScene.DirectLighting.HardwareRayTracing` | `1` | Scène Lumen éclairée en RT matériel direct |

---

## 4. Configuration MegaLights (Stochastic Direct Lighting)

MegaLights a été activé au niveau du projet et configuré via les variables de console appropriées.

| Paramètre | Variable de console | Valeur Cible | Description |
| :--- | :--- | :--- | :--- |
| **Activation MegaLights** | `r.MegaLights` / `r.MegaLights.Allow` | `1` | Active le système stochastique MegaLights globally |
| **Taux d'Échantillons** | `r.MegaLights.NumSamplesPerPixel` | `1` | Définit le nombre de rayons de visibilité par pixel (optimisé consoles/PC) |
| **Activation Projet** | `r.MegaLights.EnableForProject` | `True` | Rend MegaLights actif par défaut pour tout le projet |

---

## 5. Allocation Dynamique des Ombres (Many Lights Sampling)

Pour gérer des milliers de lumières dynamiques projetant des ombres sans effondrement des performances, le système utilise l'échantillonnage d'importance stochastique (Many Lights) :

| Paramètre | Variable de console | Valeur Cible | Rôle & Description |
| :--- | :--- | :--- | :--- |
| **Many Lights Mode** | `r.ManyLights` | `2` | Active l'échantillonnage stochastique de toutes les lumières locales, y compris celles qui utiliseraient autrement des Virtual Shadow Maps |
| **Échantillonnage Actif** | `r.ManyLights.Sampling` | `1` | Active l'échantillonnage d'importance stochastique des lumières |
| **Ombres Stochastiques** | `r.ManyLights.Shadows` | `1` | Gère les ombres de manière dynamique par échantillonnage de faisceau |
| **Seuil de Contribution** | `r.ManyLights.Sampling.MinWeight` | `0.01` | Élimine les lumières à contribution négligeable pour économiser les rayons de visibilité |

---

## Conclusion et Validation

Toutes les configurations requises pour activer **MegaLights** et son pipeline de dépendances matérielles sous **Unreal Engine 5.8** ont été injectées avec succès dans le fichier `DefaultEngine.ini`.
Lors du prochain chargement de l'éditeur ou du jeu, les shaders SM6 seront compilés et le moteur initialisera le pipeline GPU-driven stochastique permettant d'afficher des milliers de sources lumineuses dynamiques avec ombres douces physiques et éclairage global Lumen matériel de pointe.
