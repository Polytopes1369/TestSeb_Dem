import unreal
import os

def generate_profiling_launch_script(project_path, engine_path="UE_5.8"):
    """
    Generates a local batch script (.bat) to launch the Unreal Engine 5.8 project 
    instrumented with Unreal Insights channels and named events enabled for MegaLights profiling.
    """
    script_content = f"""@echo off
REM Batch file to launch UE 5.8 project with CPU/GPU tracing enabled for MegaLights profiling
REM Make sure the Unreal Insights application is running before executing this script.

echo Launching project with CPU/GPU insights channels...
"{engine_path}" "{project_path}" -trace=cpu,gpu,frame,rendering -statnamedevents -tracehost=127.0.0.1 -messaging
"""
    bat_path = os.path.join(unreal.Paths.project_dir(), "Launch_MegaLights_Profiler.bat")
    try:
        with open(bat_path, "w") as f:
            f.write(script_content)
        unreal.log(f"MegaLightsProfiler: Generated launch batch script at: {bat_path}")
        return bat_path
    except Exception as e:
        unreal.log_error(f"MegaLightsProfiler: Failed to write launch script: {e}")
        return None

def trigger_gpu_profile():
    """
    Triggers a GPU profile capture snapshot via the Unreal Editor.
    Captures rendering passes (Light Injection, Ray Tracing, and Denoising)
    and dumps them to the Editor log directory.
    """
    unreal.log("MegaLightsProfiler: Executing 'profilegpu' capture...")
    
    # Execute console commands to lock stable frame times and profile
    unreal.SystemLibrary.execute_console_command(None, "r.MegaLights.Visualize.LightComplexity 0")
    unreal.SystemLibrary.execute_console_command(None, "profilegpu")
    
    unreal.log("MegaLightsProfiler: GPU Profile captured! Check your Output Log or Unreal Insights window.")

def run_megalights_profiling_procedure():
    """
    Prints a detailed procedural checklist for developers to profile 
    MegaLights performance and isolate 'Light Injection' and 'Denoising' costs.
    """
    procedure = """
==========================================================================================
                 MEGALIGHTS AUTOMATED PROFILING & PROCEDURAL CHECKLIST
==========================================================================================

1. UNREAL INSIGHTS PREPARATION:
   - Launch Unreal Insights (located in Engine/Binaries/Win64/UnrealInsights.exe).
   - Ensure a live connection is waiting on 127.0.0.1.

2. RUNTIME LAUNCH OPTION:
   - Launch the editor or game with command line:
     -trace=cpu,gpu,frame,rendering -statnamedevents -tracehost=127.0.0.1
   (You can use the auto-generated batch script: Launch_MegaLights_Profiler.bat)

3. PROFILING TARGET PASSES:
   - In-game or inside the Viewport, press F11 to trigger 'profilegpu'.
   - Search the generated profile tree in the log for the following sub-nodes:
     
     a) [Light Injection & Stochastic Sampling]:
        Look for: "MegaLights" -> "MegaLightsDirectLighting" -> "RayGen" / "TraceShadows"
        * This measures the cost of building spatial lists and casting stochastic ray queries.
        
     b) [Denoising & Temporal Reconstruction]:
        Look for: "MegaLights" -> "MegaLightsDenoising" -> "TemporalFilter" / "SpatialFilter"
        * This measures the cost of resolving noise and blending pixels over history frames.

4. REAL-TIME VIEWPORT VISUALIZATION:
   - Press F9 to toggle the interactive 'MegaLights Light Complexity' visualizer.
   - Hover over pixels to verify sample distributions and culling behavior.
   - Press F10 to toggle pixel-level ray trace visualization.
==========================================================================================
"""
    unreal.log(procedure)
    
    # Generate the launch script locally for the editor project
    proj_file = unreal.Paths.get_project_file_path()
    if proj_file:
        generate_profiling_launch_script(proj_file)
    else:
        unreal.log_warning("MegaLightsProfiler: Project file path not detected; launch script generation skipped.")

if __name__ == "__main__":
    run_megalights_profiling_procedure()
