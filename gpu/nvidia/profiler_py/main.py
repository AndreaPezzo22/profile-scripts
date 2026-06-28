import profiler
import math_engine
import plotter
import os

def main():

    APP_executable = os.path.expanduser("~/ACA-project/muDock/build/application/muDock")
    METRICS_FILE = os.path.expanduser("~/ACA-project/profile-scripts/gpu/nvidia/METRICS")
    
    path_protein = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb")
    path_ligand = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_ligand.mol2")
    work_dir = os.path.expanduser("~/ACA-project/muDock/data/1fkb/")

    KERNEL_TO_ANALYZE = "apply_cuda"
    ARGS_MUDOCK = [
        "--protein", path_protein,
        "--ligand", path_ligand,
        "--use", "CUDA:GPU:0"
    ]

    # Catching the profiling results
    raw_data_df = profiler.profiling_ncu(
        executable=APP_executable, 
        app_args=ARGS_MUDOCK, 
        kernel_name=KERNEL_TO_ANALYZE, 
        path_to_metrics=METRICS_FILE,
        work_dir=work_dir
    )

    if raw_data_df is not None:

        # Catching the results for the different Rooflines
        fp32_roofline_data = math_engine.fp32_roofline(raw_data_df)
        fp64_roofline_data = math_engine.fp64_roofline(raw_data_df)
        ai_roofline_data = math_engine.instruction_intensity_roofline(raw_data_df)
        shared_roofline_data = math_engine.shared_memory_roofline(raw_data_df)

        # Catching the architecture features
        specs = plotter.get_gpu_specs()

        # Sending the data to the plotter
        plotter.fp32_roofline_plot(fp32_roofline_data, specs)
        plotter.fp64_roofline_plot(fp64_roofline_data, specs)
        plotter.shared_roofline_plot(shared_roofline_data, specs)
        plotter.instruction_roofline_plot(ai_roofline_data, specs)

        # Sending data to the Plotter

        print("Completed workflow")

    else:
        print("\nThe profiler did not work well")

if __name__ == "__main__":
    main()
