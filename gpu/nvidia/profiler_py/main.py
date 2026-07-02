import profiler
import math_engine
import plotter
import subprocess
import os
import argparse
import pandas as pd
import io

def list_kernels(executable, args, work_dir, max_launches=50):
    print(f"Looking for the application's kernels: {os.path.basename(executable)}...")
    
    command = [
        "ncu", 
        "--csv", 
        "--metrics", "sm__cycles_elapsed.avg", 
        "-c", str(max_launches), 
        executable
    ] + args
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=work_dir)
        
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
    parser = argparse.ArgumentParser(description="====== Roofline Profiler ======")
    parser.add_argument("-k", "--kernel", type=str, help="Name of the kernel to profile")
    parser.add_argument("-e", "--exe", type=str, required=True, help="Path to the executable")
    parser.add_argument("-wd", "--workdir", type=str, required=True, help="Path to the work directory (cwd)")
    parser.add_argument("-d", "--detect", action="store_true", help="Detects the available kernels")
    parser.add_argument("-wu", "--WARM-UP", dest="warmup", type=int, 
                        help="Insert the number if warm-up iterations",
                        default=0)    
    parser.add_argument("-m", "--METRICS", dest="metrics", type=str,
                        help="Insert path to the metrics file",
                        default="~/ACA-project/profile-scripts/gpu/nvidia/METRICS")    

    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="Arguments to pass to the executabòe")
    
    args = parser.parse_args()
    
    METRICS_FILE = os.path.expanduser(args.metrics)

    app_args = args.app_args
    if app_args and app_args[0] == "--":
        app_args = app_args[1:]

    if args.detect:
        print(f"Looking for the application's kernels: {os.path.basename(args.exe)}...")
        kernel_list = list_kernels(args.exe, app_args, args.workdir, max_launches=50) 
        print(f"The kernel list of the application is: {kernel_list}")
        return

    if args.kernel:
        print(f"\nLaunching the profiling for kernel: '{args.kernel}'...")
        
        raw_data_df = profiler.profiling_ncu(
            executable=args.exe,       
            app_args=app_args,        
            kernel_name=args.kernel,   
            path_to_metrics=METRICS_FILE, 
            work_dir=args.workdir,
            warmup=args.warmup
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
