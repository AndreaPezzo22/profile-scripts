import subprocess
import architectures
import pandas as pd
import numpy as np
import os
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def _get_metric_value(df, column, default=0.0):
    """Return the first non-null value for a metric column."""
    if column not in df.columns:
        return default

    values = df[column].dropna()
    if values.empty:
        return default

    return float(values.iloc[0])


def _get_freq_hz(df):
    """Return the GPU clock rate in Hz from the profiled metrics."""
    freq_hz = _get_metric_value(df, 'sm__cycles_elapsed.avg.per_second', 0.0)
    return freq_hz if freq_hz > 0 else 1.41e9


def get_gpu_specs(df):
    """
    Extracts theoretical peak values directly from profiled metrics.
    Uses the peak_sustained values collected from NVIDIA metrics.
    """
    
    # Get GPU name for reference
    cmd = ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]
    try:
        output = subprocess.check_output(cmd).decode("utf-8").strip().split('\n')[0].split(', ')
        gpu_name = output[0]
        cc = float(output[1])
    except:
        gpu_name = "Unknown GPU"
        cc = 0.0

    sm_count = _get_metric_value(df, 'device__attribute_multiprocessor_count')

    print(f"Detected GPU: {gpu_name}, Compute Capability: {cc}, SM Count: {sm_count}")
    
    # Computing the Clock frequence
    freq_hz = _get_freq_hz(df)
    clock_mhz = freq_hz / 1e6

    cores_per_sm = {
        6.0: {'fp32': 64, 'fp64': 32},  # Pascal P100
        6.1: {'fp32': 128, 'fp64': 4},  # Pascal GTX 1080 / Titan Xp
        7.0: {'fp32': 64, 'fp64': 32},  # Volta V100
        7.5: {'fp32': 64, 'fp64': 2},   # Turing T4 / RTX 2000
        8.0: {'fp32': 64, 'fp64': 32},  # Ampere A100
        8.6: {'fp32': 128, 'fp64': 2},  # Ampere RTX 3000 / A40
        8.9: {'fp32': 128, 'fp64': 2},  # Ada Lovelace RTX 4000
        9.0: {'fp32': 128, 'fp64': 64}  # Hopper H100
    }

    fp32_cores = cores_per_sm.get(cc, {}).get('fp32', 0)
    fp64_cores = cores_per_sm.get(cc, {}).get('fp64', 0)

    theoretical_peak_fp32 = (sm_count * fp32_cores * 2 * freq_hz) / 1e9 if fp32_cores > 0 else 0
    theoretical_peak_fp64 = (sm_count * fp64_cores * 2 * freq_hz) / 1e9 if fp64_cores > 0 else 0

    # Computing the theoretical peak of the FP32 operations
    fp32_ffma = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained')
    fp32_fmul = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_fmul_pred_on.sum.peak_sustained')
    fp32_fadd = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_fadd_pred_on.sum.peak_sustained')

    peak_gflops_fp32 = ((2 * fp32_ffma) + fp32_fmul + fp32_fadd) * freq_hz / 1e9
    if theoretical_peak_fp32 > 0 and (peak_gflops_fp32 == 0 or peak_gflops_fp32 > theoretical_peak_fp32 * 1.1):
        peak_gflops_fp32 = theoretical_peak_fp32

    # Computing the theoretical peak of the FP64 operations
    fp64_dfma = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained')
    fp64_dmul = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_dmul_pred_on.sum.peak_sustained')
    fp64_dadd = _get_metric_value(df, 'sm__sass_thread_inst_executed_op_dadd_pred_on.sum.peak_sustained')

    peak_gflops_fp64 = ((2 * fp64_dfma) + fp64_dmul + fp64_dadd) * freq_hz / 1e9
    if theoretical_peak_fp64 > 0 and (peak_gflops_fp64 == 0 or peak_gflops_fp64 > theoretical_peak_fp64 * 1.1):
        peak_gflops_fp64 = theoretical_peak_fp64

    # Computing the throughput transforming the theorretical value from memory per cycle to memory per second
    hbm_cycles = _get_metric_value(df, 'dram__bytes.sum.peak_sustained')
    bw_hbm = (hbm_cycles * freq_hz) / 1e9 if hbm_cycles > 0 else 1555.0

    l1_cycles = _get_metric_value(df, 'l1tex__t_bytes.sum.peak_sustained')
    bw_l1 = (l1_cycles * freq_hz) / 1e9 if l1_cycles > 0 else 15161.8

    l2_cycles = _get_metric_value(df, 'lts__t_bytes.sum.peak_sustained')
    bw_l2 = (l2_cycles * freq_hz) / 1e9 if l2_cycles > 0 else 5373.0

    peak_instr_cycles = _get_metric_value(df, 'sm__sass_thread_inst_executed_total.sum.peak_sustained')
    peak_gips = (peak_instr_cycles * freq_hz) / 1e9 if peak_instr_cycles > 0 else (sm_count * 4 * clock_mhz / 1000)

    shared_ld = _get_metric_value(df, 'l1tex__data_pipe_l1_op_shared_ld_m_bytes.sum.peak_sustained')
    shared_st = _get_metric_value(df, 'l1tex__data_pipe_l1_op_shared_st_m_bytes.sum.peak_sustained')

    shared_bw = ((shared_ld + shared_st) * freq_hz) / 1e9
    if shared_bw == 0:
        shared_bw = sm_count * 32 * 4 * (clock_mhz / 1000)

    # Shared memory transactions per second (assuming 32 bytes per transaction)
    trans_bw = shared_bw / 32.0 if shared_bw > 0 else peak_gips
    
    print(f"\n[GPU SPECS from Metrics]")
    print(f"  GPU: {gpu_name}")
    print(f"  FP32 Peak: {peak_gflops_fp32:.1f} GFLOP/s")
    print(f"  FP64 Peak: {peak_gflops_fp64:.1f} GFLOP/s")
    print(f"  Peak GIPS: {peak_gips:.1f} GIPS")
    print(f"  L1 BW: {bw_l1:.1f} GB/s")
    print(f"  L2 BW: {bw_l2:.1f} GB/s")
    print(f"  HBM BW: {bw_hbm:.1f} GB/s")
    print(f"  Shared BW: {shared_bw:.1f} GB/s\n")

    return {
        "gpu_name": gpu_name,
        "peak_fp32": peak_gflops_fp32,
        "peak_fp64": peak_gflops_fp64,
        "bandwidth": bw_hbm,
        "bw_l2": bw_l2,                      
        "bw_l1": bw_l1,
        "peak_gips": peak_gips,              
        "shared_bw": shared_bw,        
        "trans_bw": trans_bw
    }

def fp32_roofline_plot(results, specs, plot_dir):
    print("Generating the FP32 plot")
    
    # Catching the theoretical values
    peak = specs["peak_fp32"]
    bw_dict = {
        'L1': specs["bw_l1"],
        'L2': specs["bw_l2"],
        'HBM': specs["bandwidth"]
    }
   
   # Creating the plot as log scale
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 2e2)
    ax.set_ylim(0.1, 1e5) 

    x_vals = np.logspace(-3, 3, 500)
    ax.axhline(peak, color='black', linewidth=1.2)
    ax.text(1e1, peak * 1.1, f"Theoretical peak FP32: {peak:.1f} GFLOP/s", fontsize=16, ha='center')

    colors = {'L1': '#D49A2A', 'L2': '#2A9D8F', 'HBM': '#6D4C41'}
    markers = {'L1': 'o', 'L2': '^', 'HBM': 's'}

    # Drawing the different memory level L1, L2, HBM
    for mem_level, bw in bw_dict.items():
        color = colors.get(mem_level, 'blue')
        # For each intensity level x, computes expected performance
        y_vals = x_vals * bw
        
        # Plots the valid ideas drawing the memory line
        valid_idx = y_vals <= peak
        ax.plot(x_vals[valid_idx], y_vals[valid_idx], color=color, linewidth=1.2)
        
        # Finds the Ridge point, which is the meeting point between the peak and the bandwidth
        ridge_x = peak / bw
        label_x = ridge_x / 3
        label_y = label_x * bw
        ax.text(label_x/3, label_y * 0.2, f"{mem_level} {bw:.1f} GB/s", 
                color=color, rotation=38, fontsize=14, ha='center', va='bottom')

    perf = results['Performance (GFLOP/s)']

    print(f"  [Debug] FP32 GFLOP/s: {perf:.1f}")
    
    x_keys_dict = {
        'L1': 'L1 Arithmetic Intensity (FLOP/B)',
        'L2': 'L2 Arithmetic Intensity (FLOP/B)',
        'HBM': 'HBM Arithmetic Intensity (FLOP/B)'
    }
    
    for mem_level, x_key in x_keys_dict.items():
        if x_key in results:
            valore_ai = results[x_key]
            print(f"  [Debug] {x_key}: {valore_ai}" )
            if valore_ai > 0:
                color = colors.get(mem_level, 'red')
                marker = markers.get(mem_level, 'o')
                # Creates the point on the plot for the corresponfing memory level
                ax.scatter(valore_ai, perf, color=color, marker=marker, 
                           s=150, zorder=5, label=f"Kernel ({mem_level})")

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)

    ax.set_xlabel('Arithmetic Intensity (FLOP/B)', fontsize=14)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=14)
    ax.set_title("Hierarchical Roofline Model FP32", fontsize=14)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roofline_fp32.pdf"), format='pdf')
    plt.close()
    print("✅ Plot roofline_fp32.pdf saved") 

def fp64_roofline_plot(results, specs, plot_dir):
    print("Generating the FP64 Hierarchical plot")
    
    peak = specs["peak_fp64"]
    bw_dict = {
        'L1': specs["bw_l1"],
        'L2': specs["bw_l2"],
        'HBM': specs["bandwidth"]
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 2e2)
    ax.set_ylim(0.1, 1e5) 

    x_vals = np.logspace(-3, 3, 500)

    ax.axhline(peak, color='black', linewidth=1.2)
    ax.text(1e1, peak * 1.1, f"Theoretical peak FP64: {peak:.1f} GFLOP/s", fontsize=16, ha='center')

    colors = {'L1': '#D49A2A', 'L2': '#2A9D8F', 'HBM': '#6D4C41'}
    markers = {'L1': 'o', 'L2': '^', 'HBM': 's'}

    for mem_level, bw in bw_dict.items():
        color = colors.get(mem_level, 'blue')
        y_vals = x_vals * bw
        
        valid_idx = y_vals <= peak
        ax.plot(x_vals[valid_idx], y_vals[valid_idx], color=color, linewidth=1.2)
        
        ridge_x = peak / bw
        label_x = ridge_x / 3  # Posiziona a 1/3 della salita
        label_y = label_x * bw
        ax.text(label_x/3, label_y*0.2, f"{mem_level} {bw:.1f} GB/s", 
                color=color, rotation=38, fontsize=14, ha='center', va='bottom')

    perf = results['Performance (GFLOP/s)']
    
    x_keys_dict = {
        'L1': 'L1 Arithmetic Intensity (FLOP/B)',
        'L2': 'L2 Arithmetic Intensity (FLOP/B)',
        'HBM': 'HBM Arithmetic Intensity (FLOP/B)'
    }

    print(f"  [Debug] FP64 GFLOP/s: {perf:.1f}")
    
    points_plotted = False
    
    for mem_level, x_key in x_keys_dict.items():
        if x_key in results:
            valore_ai = results[x_key]
            print(f"  [Debug] {x_key}: {valore_ai}" )

            if pd.notna(valore_ai) and valore_ai > 0: 
                color = colors.get(mem_level, 'red')
                marker = markers.get(mem_level, 'o')
                ax.scatter(valore_ai, perf, color=color, marker=marker, 
                           s=150, zorder=5, label=f"Kernel ({mem_level})")
                points_plotted = True

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)

    ax.set_xlabel('Arithmetic Intensity (FLOP/B)', fontsize=14)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=14)
    ax.set_title("Hierarchical Roofline Model FP64", fontsize=14)
    
    if points_plotted:
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roofline_fp64.pdf"), format='pdf')
    plt.close()
    print("✅ Plot roofline_fp64.pdf saved")


def instruction_roofline_plot(results, specs, plot_dir):

    print("Generating the Instruction Hierarchical plot")
    
    peak_gips = specs["peak_gips"]
    
    bw_dict_bytes = {
        'L1': specs.get("bw_l1"),
        'L2': specs.get("bw_l2"),
        'HBM': specs.get("bandwidth")
    }
    
    # We transform the bandwidth values from byte/s to transactions/s.
    # We assume that the size of a single transaction for global memory is is 32 bytes (E' corretto ?)
    tx_dict = { mem_level: bw / 32.0 for mem_level, bw in bw_dict_bytes.items() }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(1e0, 1e3)

    ax.hlines(y=peak_gips, xmin=1e-2, xmax=2e3, color='black', linewidth=1.2)
    ax.text(1e1, peak_gips * 1.1, f"Theoretical peak: {peak_gips:.1f} warp GIPS", fontsize=16, ha='center')

    colors = {'L1': 'red', 'L2': 'lime', 'HBM': 'blue'}

    for mem_level, tx_rate in tx_dict.items():
        color = colors.get(mem_level, 'black')
        
        ridge_x = peak_gips / tx_rate
        bottom_x = 1e-3 / tx_rate
        
        ax.plot([bottom_x, ridge_x], [1e-3, peak_gips], color=color, linewidth=1.2)
        
        label_x = ridge_x / 3
        label_y = label_x * tx_rate
        ax.text(label_x/3, label_y * 0.2, f"{mem_level} {tx_rate:.1f} GTXN/s", 
                color=color, rotation=42, fontsize=14, ha='center', va='bottom')

    perf = results.get('Performance GIPS')
    valore_ai = results.get('Instruction Intensity')
    
    print(f"  [Debug] Drawinf Kernel at: X={valore_ai:.2f}, Y={perf:.2f}")
    
    points_plotted = False
    if perf > 0 and valore_ai > 0 and not math.isnan(perf) and not math.isnan(valore_ai):
        ax.scatter(valore_ai, perf, color='red', marker='s', 
                   s=150, zorder=5, label="Kernel")
        points_plotted = True

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)

    ax.set_xlabel('Instruction Intensity (warp instructions per transaction)', fontsize=14)
    ax.set_ylabel('Performance (warp GIPS)', fontsize=14)
    
    if points_plotted:
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "roofline_instructions.pdf"), format='pdf')
    plt.close(fig)
    print("✅ Plot roofline_instructions.pdf saved")

def shared_roofline_plot(results, specs, plot_dir):
    print("Generating the Shared Memory plot")
    
    # Theoretical peak values
    peak_gips = specs.get("peak_gips", 600.0)
    max_tx_rate = specs.get("trans_bw", peak_gips)
  
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(1e0, 1e3) 

    color_shared = 'magenta'

    ax.hlines(y=peak_gips, xmin=1e-2, xmax=2e3, color='black', linewidth=1.2)
    ax.text(1e1, peak_gips * 1.1, f"Theoretical peak: {peak_gips:.1f} warp GIPS", ha='center', fontsize=16)

    ridge_x = peak_gips / max_tx_rate
    ax.plot([1e-3, ridge_x], [1e-3 * max_tx_rate, peak_gips], color=color_shared, linewidth=1.2)

    label_x = ridge_x / 10
    label_y = label_x * max_tx_rate
    ax.text(label_x, label_y * 1.1, f"Shared {max_tx_rate:.1f} GTXN/s", 
            color=color_shared, rotation=45, fontsize=10, ha='center', va='bottom')

    # The Shared memory is organized in in banks, if more threads of the same warp access to the same bank
    # in the same cycle, a conflict occurs. In that case, the systems needs to serialize the accesses .
    # 

    # A) No bank conflict (x = 1.0)
    y_no_conflict = min(1.0 * max_tx_rate, peak_gips)
    ax.vlines(x=1.0, ymin=1e0, ymax=y_no_conflict, color=color_shared, linewidth=1.0)
    ax.text(1.15, 1.2, "No bank conflict", color=color_shared, rotation=90, va='bottom', ha='left', fontsize=14)

    # B) 32-way bank conflict (x = 1/32)
    x_conflict = 1.0 / 32.0
    y_conflict_max = min(x_conflict * max_tx_rate, peak_gips)
    ax.vlines(x=x_conflict, ymin=1e0, ymax=y_conflict_max, color=color_shared, linewidth=1.0)
    ax.text(x_conflict * 1.15, 1.2, "32-way bank conflict", color=color_shared, rotation=90, va='bottom', ha='left', fontsize=14)

    perf = results.get('Performance GIPS Shared')
    valore_ai = results.get('Shared Intensity')

    print(f"  [Debug] Shared Performance (GIPS): {perf:.2f}")
    print(f"  [Debug] Shared Intensity (warp instr/tx): {valore_ai:.2f}")
    
    if perf > 0 and valore_ai > 0 and not math.isnan(perf) and not math.isnan(valore_ai):
        ax.scatter(valore_ai, perf, color=color_shared, marker='s', s=60, zorder=5)
        ax.text(valore_ai * 0.85, perf, color='black', ha='right', va='center', fontsize=14)

    ax.grid(True, which="major", ls="--", color="black", alpha=0.5)
    ax.grid(True, which="minor", ls=":", color="black", alpha=0.5)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.set_xlabel('Instruction Intensity (warp instructions per transaction)', fontsize=14)
    ax.set_ylabel('Performance (warp GIPS)', fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "roofline_shared.pdf"), format='pdf')
    plt.close(fig)
    print("✅ Plot roofline_shared.pdf saved")
