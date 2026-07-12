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
    
    # Building the command
    command = [
        "ncu", 
        "--csv", 
        "--metrics", "sm__cycles_elapsed.avg", 
        "-c", str(max_launches), 
        executable
    ] + args
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=work_dir)
        
        # We select all the rows of the output and we keep just those referred to the kernels
        csv_lines = [line for line in result.stdout.split('\n') if line.startswith('"ID"') or (line and line.startswith('"') and "Process ID" not in line)]
        
        if len(csv_lines) <= 1:
            print("No Kernel found")
            return []
        
        # Creates the Data Frame Pandas
        testo_csv = '\n'.join(csv_lines) # The lines are displaced in a single one
        df = pd.read_csv(io.StringIO(testo_csv), thousands=',' ) # thousands handles the thousands in the file
        df.columns = df.columns.str.replace('"', '').str.strip() # Cleans the columns names from "" and additional spaces
        df = df.drop(0).reset_index(drop=True) # Erases the first row containing the metadata
        
        # Extracts pure kernel names without < and ( symbols
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

    # Collects the arguments needed by the application to run 
    parser = argparse.ArgumentParser(description="====== Roofline Profiler ======")
    parser.add_argument("-k", "--kernel", type=str, 
                        help="Name of the kernel to profile")
    parser.add_argument("-e", "--exe", type=str, required=True, 
                        help="Path to the executable")
    parser.add_argument("-wd", "--workdir", type=str, required=True, 
                        help="Path to the work directory (cwd)")
    parser.add_argument("-d", "--detect", action="store_true", 
                        help="Detects the available kernels")
    parser.add_argument("-wu", "--WARM-UP", dest="warmup", type=int, 
                        help="Insert the number if warm-up iterations",
                        default=0)    
    parser.add_argument("-m", "--METRICS", dest="metrics", type=str,
                        help="Insert path to the metrics file",
                        default="~/ACA-project/profile-scripts/gpu/nvidia/METRICS")    

    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="Arguments to pass to the executabòe")
    
    args = parser.parse_args()
    
    METRICS_FILE = os.path.expanduser(args.metrics)

    # All the elements after -- are selected as arguments
    app_args = args.app_args
    if app_args and app_args[0] == "--":
        app_args = app_args[1:]

    # If the -d flag is selected, the application's kernel list is returned
    if args.detect:
        kernel_list = list_kernels(args.exe, app_args, args.workdir, max_launches=50) 
        print(f"The kernel list of the application is: {kernel_list}")
        return

    # Otherwise, the normal profilation is executed
    if args.kernel:
        
        # All the collected data get sent to the profiler
        raw_data_df = profiler.profiling_ncu(
            executable=args.exe,       
            app_args=app_args,        
            kernel_name=args.kernel,   
            path_to_metrics=METRICS_FILE, 
            work_dir=args.workdir,
            warmup=args.warmup
        )

    if raw_data_df is not None:

        # Creating a folder to save the plots where we are executing the script
        plot_dir = os.path.join(os.getcwd(), f"roofline_plots_{args.exe.split(os.sep)[-1].split('.')[0]}_{args.kernel}")
        os.makedirs(plot_dir, exist_ok=True)

        # Catching the results for the different Rooflines
        fp32_roofline_data = math_engine.fp32_roofline(raw_data_df)
        fp64_roofline_data = math_engine.fp64_roofline(raw_data_df)
        ai_roofline_data = math_engine.instruction_intensity_roofline(raw_data_df)
        shared_roofline_data = math_engine.shared_memory_roofline(raw_data_df)

        # Catching the architecture features
        specs = plotter.get_gpu_specs(raw_data_df)

        # Sending the data to the plotter
        plotter.fp32_roofline_plot(fp32_roofline_data, specs, plot_dir)
        plotter.fp64_roofline_plot(fp64_roofline_data, specs, plot_dir)
        plotter.shared_roofline_plot(shared_roofline_data, specs, plot_dir)
        plotter.instruction_roofline_plot(ai_roofline_data, specs, plot_dir)

        # Sending data to the Plotter

        print("Completed workflow")

    else:
        print("\nThe profiler did not work well")

if __name__ == "__main__":
    main()
