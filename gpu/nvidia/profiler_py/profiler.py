import subprocess
import pandas as pd
import os
import shutil
import io

def detect_architecture():
    """
    Detects the architecture of the NVIDIA GPU.
    """

    print("Detecting the architecture")

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

    print("Building the Metrics line")

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

    # HBM Bandwidth
    hbm_metrics = [
        "dram__bytes.sum.peak_sustained"
    ]
    
    # L1 and L2 Cache Bandwidth
    cache_metrics = [
        "l1tex__t_bytes.sum.peak_sustained",  # L1 cache bandwidth
        "lts__t_bytes.sum.peak_sustained"      # L2 cache bandwidth
    ]
    
    # Shared Memory Bandwidth
    shared_metrics = [
        "l1tex__data_pipe_l1_op_shared_ld_m_bytes.sum.peak_sustained",  # Shared load
        "l1tex__data_pipe_l1_op_shared_st_m_bytes.sum.peak_sustained"   # Shared store
    ]
    
    # FP32 Operations (single precision)
    fp32_metrics = [
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",  # FFMA (fused multiply-add)
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum.peak_sustained",  # FMUL
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum.peak_sustained"   # FADD
    ]
    
    # FP64 Operations (double precision)
    fp64_metrics = [
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",  # DFMA (fused multiply-add)
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum.peak_sustained",  # DMUL
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum.peak_sustained"   # DADD
    ]
    
    # Instructions and Cycles
    instruction_metrics = [
        "sm__sass_thread_inst_executed_total.sum.peak_sustained",  # Total instructions
        "sm__cycles_elapsed.avg.per_second"                         # Cycles per second
    ]
    
    # Combine all theoretical metrics
    theoretical_metrics = (hbm_metrics + cache_metrics + shared_metrics + 
                          fp32_metrics + fp64_metrics + instruction_metrics)

    # Uniamo tutto
    valid_metrics.extend(theoretical_metrics)
    valid_metrics.extend([l1_peak, l1_per_sec])
    # Removing duplicates
    valid_metrics = list(dict.fromkeys(valid_metrics))
    
    print(f"Metrics loaded succesfully {len(valid_metrics)}.")
    return ",".join(valid_metrics)


def profiling_ncu(executable, app_args, kernel_name, path_to_metrics, work_dir, warmup):
    """
    Launching Nsight Compute, running the app and creating the CSV file
    """
    print(f"\nLaunching the profiling for kernel: '{kernel_name}'...")

    gpu_architecture = detect_architecture()
    metrics_string = build_metrics_string(path_to_metrics, gpu_architecture)    
    if not metrics_string:
        print("\nERROR. No metric found.")
        return None
    
    if not gpu_architecture:
        print("\nERROR. No architecture found.")

    command = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--kernel-name", kernel_name,
        "--metrics", metrics_string,
        executable
    ]
    command.extend(app_args)
    
    if warmup !=0:
        print("Inizio warmup...")
        for _ in range(warmup):
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Launching NCU and catching the results
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
        # Creates the csv just with the lines containing the profiling values    
        if is_csv and linea.strip():
            csv_lines.append(linea)

    if not csv_lines:
        print("Impossible to find the csv:")
        print(result.stderr)
        return None

    text_csv = '\n'.join(csv_lines)

    # The csv get saved in the current directory
    path_csv = os.path.join(os.getcwd(), f"report_ncu_{kernel_name}.csv")
    with open(path_csv, "w") as f:
        f.write(text_csv)
    
    print(f"The csv has been generated and saved in: {path_csv}")

    df = pd.read_csv(io.StringIO(text_csv), thousands=',')
    df.columns = df.columns.str.replace('"', '').str.strip()

    df = df.drop(0).reset_index(drop=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
