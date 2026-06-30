import profiler
import math_engine
import plotter
import subprocess
import os
import pandas as pd
import io

def list_kernels(executable, args, max_launches=50):
    print(f"Looking for the application's kernels: {os.path.basename(executable)}...")
    
    command = [
        "ncu", 
        "--csv", 
        "--metrics", "sm__cycles_elapsed.avg", 
        "-c", str(max_launches), 
        executable
    ] + args
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        csv_lines = [line for line in result.stdout.split('\n') if line.startswith('"ID"') or (line and line.startswith('"') and "Process ID" not in line)]
        
        if len(csv_lines) <= 1:
            print("No Kernel found")
            return []
            
        testo_csv = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(testo_csv), thousands=',')
        df.columns = df.columns.str.replace('"', '').str.strip()
        df = df.drop(0).reset_index(drop=True)
        
        if 'Kernel Name' in df.columns:
            
            kernels_grezzi = df['Kernel Name'].dropna().unique()
            
            kernels_puliti = set()
            for k in kernels_grezzi:
                nome_base = k.split('<')[0].split('(')[0].strip()
                kernels_puliti.add(nome_base)
                
            return sorted(list(kernels_puliti))
        else:
            return []

    except Exception as e:
        print(f"Error: {e}")
        return []
    

def main():

    APP_executable = os.path.expanduser("~/ACA-project/muDock/build/application/muDock")
    METRICS_FILE = os.path.expanduser("~/ACA-project/profile-scripts/gpu/nvidia/METRICS")
    
    path_protein = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb")
    path_ligand = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_ligand.mol2")
    work_dir = os.path.expanduser("~/ACA-project/muDock/data/1fkb/")

    KERNEL_TO_ANALYZE = "calc_energy"
    ARGS_MUDOCK = [
        "--protein", path_protein,
        "--ligand", path_ligand,
        "--use", "CUDA:GPU:0"
    ]

    print(f"The kernel list of the application is: {list_kernels(APP_executable, ARGS_MUDOCK)}")

    # return None

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
