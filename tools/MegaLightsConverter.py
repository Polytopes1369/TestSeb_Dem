import unreal

def convert_local_lights_to_megalights(folder_path=None, max_radius_clamp=1200.0, min_radius_limit=200.0):
    """
    Scans the active level for Point, Spot, and Rect lights, converts them to the MegaLights pipeline,
    disables obsolete shadow methods, and optimizes their attenuation radii to prevent BVH and culling overhead.
    
    Args:
        folder_path (str/Name): Optional World Outliner folder path to restrict the scan (e.g. "/Lights/Local").
        max_radius_clamp (float): Maximum allowed attenuation radius in Unreal units to prevent excessive overlap.
        min_radius_limit (float): Minimum bound for attenuation radius to prevent over-constricting lights.
    """
    # 1. Get the Editor Actor Subsystem to query actors in the active level
    editor_actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not editor_actor_subsystem:
        unreal.log_error("MegaLightsConverter: EditorActorSubsystem not found. Make sure you are running in the Editor context.")
        return
        
    all_actors = editor_actor_subsystem.get_all_level_actors()
    
    converted_count = 0
    total_radius_reduction = 0.0
    report_details = []
    
    unreal.log("MegaLightsConverter: Starting scan of active level...")
    if folder_path:
        unreal.log(f"MegaLightsConverter: Filtering by Outliner folder path: '{folder_path}'")

    # 2. Iterate over all actors in the level
    for actor in all_actors:
        # If folder path filter is specified, check actor's folder
        if folder_path:
            actor_folder = str(actor.get_folder_path())
            if not actor_folder.startswith(str(folder_path)):
                continue
                
        # Find light components on the actor
        # Specifically target Point, Spot, and Rect lights (Local lights)
        light_components = actor.get_components_by_class(unreal.LightComponent)
        for comp in light_components:
            # Skip Directional Lights (not local lights) or Sky Lights
            if isinstance(comp, (unreal.DirectionalLightComponent, unreal.SkyLightComponent)):
                continue
                
            # We are interested in Point, Spot, and Rect lights
            if not isinstance(comp, (unreal.PointLightComponent, unreal.SpotLightComponent, unreal.RectLightComponent)):
                continue
                
            # Check mobility - MegaLights is meant for dynamic (Movable) or Stationary lights
            mobility = comp.get_editor_property("mobility")
            if mobility not in [unreal.ComponentMobility.MOVABLE, unreal.ComponentMobility.STATIONARY]:
                continue
                
            actor_name = actor.get_actor_label()
            comp_name = comp.get_name()
            
            # --- 3. Enable MegaLights Properties ---
            # Enable Allow MegaLights
            comp.set_editor_property("allow_mega_lights", True)
            
            # Set the MegaLights shadow method to Ray Tracing for stochastic area shadowing
            # This uses the modern GPU-driven BVH rather than costly Virtual Shadow Maps
            try:
                comp.set_editor_property(
                    "mega_lights_shadow_method", 
                    unreal.MegaLightsShadowMethod.RAY_TRACING
                )
                shadow_method_status = "Ray Tracing"
            except Exception as e:
                unreal.log_warning(f"MegaLightsConverter: Could not set mega_lights_shadow_method on {comp_name}. Error: {e}")
                shadow_method_status = "Default/Error"
                
            # --- 4. Disable Obsolete Shadow Methods ---
            # Disable legacy Ray Traced Distance Field Shadows
            had_df_shadows = comp.get_editor_property("use_ray_traced_distance_field_shadows")
            comp.set_editor_property("use_ray_traced_distance_field_shadows", False)
            
            # Disable legacy translucent shadow casting (redundant under MegaLights)
            had_translucent_shadows = comp.get_editor_property("cast_translucent_shadows")
            comp.set_editor_property("cast_translucent_shadows", False)
            
            # Disable legacy volumetric shadow casting if needed
            had_volumetric_shadows = comp.get_editor_property("cast_volumetric_shadow")
            comp.set_editor_property("cast_volumetric_shadow", False)
            
            # --- 5. Optimize Attenuation Radius for BVH Culling ---
            # Get current attenuation radius
            orig_radius = comp.get_editor_property("attenuation_radius")
            intensity = comp.get_editor_property("intensity")
            
            # Heuristic calculation for optimized radius based on light intensity:
            # We want to avoid massive overlapping volumes that flood the grid culling structure.
            # In Unreal, 1 Lumen = 1 / 4pi Candelas. Point lights use Lumens, while Spot/Rect lights use Candelas.
            # For PointLights (Lumens): intensity is high (e.g. 5000 lm).
            # We determine a target radius proportional to the square root of intensity, clamped to our limits.
            if isinstance(comp, unreal.PointLightComponent):
                # Point Light: Estimate radius scaling
                target_radius = orig_radius * 0.75 # Default scale reduction
                # Let's align it with intensity:
                calculated_radius = (intensity ** 0.5) * 8.0
                target_radius = min(target_radius, calculated_radius)
            else:
                # Spot and Rect Lights: Tighter cones mean less physical overlap, but we still clamp
                target_radius = orig_radius * 0.85
                calculated_radius = (intensity ** 0.5) * 12.0
                target_radius = min(target_radius, calculated_radius)
                
            # Apply clamps
            optimized_radius = max(min_radius_limit, min(target_radius, max_radius_clamp, orig_radius))
            comp.set_editor_property("attenuation_radius", optimized_radius)
            
            radius_reduction = orig_radius - optimized_radius
            total_radius_reduction += radius_reduction
            converted_count += 1
            
            # Save stats for the summary report
            report_details.append({
                "actor": actor_name,
                "type": type(comp).__name__,
                "mobility": "Movable" if mobility == unreal.ComponentMobility.MOVABLE else "Stationary",
                "intensity": intensity,
                "orig_radius": orig_radius,
                "opt_radius": optimized_radius,
                "reduction": radius_reduction,
                "shadow_method": shadow_method_status
            })
            
            # Mark the actor as dirty so the editor knows it changed and needs saving
            actor.modify()

    # 6. Print Report Summary
    unreal.log("==========================================================================================")
    unreal.log("                    MEGALIGHTS CONVERSION & OPTIMIZATION REPORT                           ")
    unreal.log("==========================================================================================")
    unreal.log(f"Scan Scope: {'Folder: ' + str(folder_path) if folder_path else 'Entire Level'}")
    unreal.log(f"Total Local Lights Converted to MegaLights: {converted_count}")
    unreal.log(f"Total Attenuation Radius Reduction: {total_radius_reduction:.2f} Unreal Units")
    unreal.log("------------------------------------------------------------------------------------------")
    unreal.log(f"{'Actor Name':<30} | {'Light Type':<20} | {'Mobility':<10} | {'Orig Rad':<10} | {'Opt Rad':<10} | {'Reduction':<10}")
    unreal.log("------------------------------------------------------------------------------------------")
    for detail in report_details:
        unreal.log(f"{detail['actor']:<30} | {detail['type']:<20} | {detail['mobility']:<10} | {detail['orig_radius']:<10.1f} | {detail['opt_radius']:<10.1f} | {detail['reduction']:<10.1f}")
    unreal.log("==========================================================================================")
    
    return report_details

# Execution entry point (can be executed in the editor python console)
if __name__ == "__main__":
    # Convert all local lights in the level with default optimization constraints
    convert_local_lights_to_megalights()
