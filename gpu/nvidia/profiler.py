import subprocess
import pandas as pd
import os
import shutil
import io

def detect_architecture():
    """
    Detects the architecture of the NVIDIA GPU.
    """
    try:
        # This command query the architecture to know the model of the device
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        stdout = (result.stdout or "").strip()
        if not stdout:
            raise ValueError("Empty output from nvidia-smi")
        
        # If the server has more than one GPU we return just the first one
        cc_stringa = stdout.split('\n')[0]
        architecture = int(float(cc_stringa) * 10)
        print(f"The GPU architecture is: {architecture}")
        return architecture
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Impossible to retrieve the GPU architecture ({e}). Uso fallback: 80")
        return 80

def build_metrics_string(path_to_metrics, architecture):
    """
    Reads the file METRICS and build the string of metrics that will be used for profiling.
    """
    if not os.path.exists(path_to_metrics):
        print(f"Metrics file NOT FOUND: {path_to_metrics}")
        return ""

    # Metrics of the L1 cache to exclude for incompatibility hardware
    l1_to_exclude = [
        "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained",
        "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained",
        "l1tex__lsu_writeback_active.sum.peak_sustained",
        "l1tex__lsu_writeback_active_mem_lgds.sum.per_second",
        "l1tex__lsu_writeback_active_mem_lg.sum.per_second",
        "l1tex__lsu_writeback_active.sum.per_second"
    ]

    valid_metrics = []
    
    with open(path_to_metrics, 'r') as f:
        for each_line in f:
            # Removing the comments lines
            metric = each_line.split('#')[0].strip()
            if metric and metric not in l1_to_exclude:
                valid_metrics.append(metric)

    if architecture >= 90:
	l1_peak = "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active_mem_lgds.sum.per_second"
    elif architecture in [75, 80, 86, 87, 88, 89]:
        l1_peak = "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active_mem_lg.sum.per_second"
    else:
        l1_peak = "l1tex__lsu_writeback_active.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active.sum.per_second"

    valid_metrics.extend([l1_peak, l1_per_sec])
    # Removing duplicates
    valid_metrics = list(dict.fromkeys(valid_metrics))
    
    print(f"Metrics loaded succesfully {len(valid_metrics)}.")
    return ",".join(valid_metrics)


def profiling_ncu(executable, app_args, kernel_name, path_to_metrics, work_dir):
    """
    Launching Nsight Compute, running the app and creating the CSV file
    """
    print(f"\nLaunching the profiling for kernel: '{kernel_name}'...")

    gpu_architecture = detect_architecture()
    metrics_string = build_metrics_string(path_to_metrics, gpu_architecture)    
    if not metrics_string:
        print("\nNo metric found.")
        return None

    command = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--kernel-name", kernel_name,
        "--metrics", metrics_string,
        executable
    ]
    command.extend(app_args)
    
    # print(f"The full command is : {command}")

    try:
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200, cwd=work_dir)
    except Exception as e:
        print(f"Critical error during the execution: {e}")
        return None

    # Extracting the csv from the text
    output_ncu = result.stdout
    csv_lines = []
    is_csv = False

    for linea in output_ncu.split('\n'):
        # NCU starts the csv always with ID
        if linea.startswith('"ID"'):
            is_csv = True
            
        if is_csv and linea.strip():
            csv_lines.append(linea)

    if not csv_lines:
        print("Impossible to find the csv:")
        print(result.stderr)
        return None

    text_csv = '\n'.join(csv_lines)

    # The csv get saved in the current directory
    path_csv = os.path.join(os.getcwd(), "report_ncu_{kernel_name}.csv")
    with open(path_csv, "w") as f:
        f.write(text_csv)
    
    print(f"The csv has been generated and saved in: {path_csv}")

    df = pd.read_csv(io.StringIO(text_csv))
    print("Data loaded in the data frame!")
    return df


if __name__ == "__main__":

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

    # Starting the pipeline
    raw_data_df = profiling_ncu(
        executable=APP_executable, 
        app_args=ARGS_MUDOCK, 
        kernel_name=KERNEL_TO_ANALYZE, 
        path_to_metrics=METRICS_FILE,
        work_dir=work_dir
    )

    # Printing the data for a fast check
    if raw_data_df is not None:
        print("\n" + "="*50)
        print("PRINTING THE FIRST FEW LINES")
        print("="*50)

        raw_data_df.columns = raw_data_df.columns.str.replace('"', '').str.strip()

        colonne_reali = list(raw_data_df.columns)
        print(f"Total amount of metrics are (columns): {len(colonne_reali)}")

        print("\n--- The data are ---")
        print(raw_data_df.iloc[:, :5].head())
        print("="*50 + "\n")
