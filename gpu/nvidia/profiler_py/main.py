import profiler
import math_engine
import plotter
import subprocess
import os
import argparse
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

    parser = argparse.ArgumentParser(description="Application profiler") 

    parser.add_argument("-d", "--description", action="store_true", 
                        help="Pass this flag if you want to list the available kernels")
    
    # Usa dest="..." per dire ad argparse come chiamare la variabile dentro "args"
    parser.add_argument("-e", "--EXE", dest="executable", type=str, 
                        help="Insert the path to the executable", 
                        default="~/ACA-project/muDock/build/application/muDock")
    
    parser.add_argument("-m", "--METRICS", dest="metrics", type=str, 
                        help="Insert path to the metrics file", 
                        default="~/ACA-project/profile-scripts/gpu/nvidia/METRICS")
    
    parser.add_argument("-l", "--LIGAND", dest="ligand", type=str, 
                        help="Insert the path to the ligand necessary for the application", 
                        default="~/ACA-project/muDock/data/1fkb/1fkb_ligand.mol2")
    
    parser.add_argument("-p", "--PROTEIN", dest="protein", type=str, 
                        help="Insert the path to the pocket", 
                        default="~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb")
    
    parser.add_argument("-wd", "--WORK_DIR", dest="dir", type=str, 
                        help="Insert the path to the work directory", 
                        default="~/ACA-project/muDock/data/1fkb/")
    
    parser.add_argument("-k", "--KERNEL", dest="kernel", type=str, 
                        help="Insert the kernel to analyze", 
                        default="calc_energy")
    
    parser.add_argument("-wu", "--WARM-UP", dest="warmup", type=int, 
                        help="Insert the number if warm-up iterations",
                        default=0)
    
    args = parser.parse_args()


    APP_executable = os.path.expanduser(args.executable)
    METRICS_FILE = os.path.expanduser(args.metrics)
    
    path_protein = os.path.expanduser(args.protein)
    path_ligand = os.path.expanduser(args.ligand)
    work_dir = os.path.expanduser(args.dir)

    KERNEL_TO_ANALYZE = args.kernel

    warmup = args.warmup

    ARGS_MUDOCK = [
        "--protein", path_protein,
        "--ligand", path_ligand,
        "--use", "CUDA:GPU:0"
    ]

    if args.description:
        print(f"The kernel list of the application is: {list_kernels(APP_executable, ARGS_MUDOCK)}")
        return None

    # Catching the profiling results
    raw_data_df = profiler.profiling_ncu(
        executable=APP_executable, 
        app_args=ARGS_MUDOCK, 
        kernel_name=KERNEL_TO_ANALYZE, 
        path_to_metrics=METRICS_FILE,
        work_dir=work_dir
        warmup=warmup
    )

    if raw_data_df is not None:

        # Catching the results for the different Rooflines
        fp32_roofline_data = math_engine.fp32_roofline(raw_data_df)
        fp64_roofline_data = math_engine.fp64_roofline(raw_data_df)
        ai_roofline_data = math_engine.instruction_intensity_roofline(raw_data_df)
        shared_roofline_data = math_engine.shared_memory_roofline(raw_data_df)

        # Catching the architecture features
        specs = plotter.get_gpu_specs(raw_data_df)

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
