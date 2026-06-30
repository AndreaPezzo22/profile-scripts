import subprocess
import architectures
import pandas as pd
import numpy as np
import os
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def get_gpu_specs():

    """
        This function computes all the theoretical values that needs to be compared with thereal values in the Model
          -  FP32
          -  FP64
          -  Shared Memory
          -  Instruction Intensity
    """
    
    cmd = ["nvidia-smi", "--query-gpu=name,clocks.max.graphics,compute_cap", "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd).decode("utf-8").strip().split(',')
    
    if len(out) < 3:
        print(f"\nOutput of the arch query gone wrong -> {out}")
        return None
    else:
        gpu_name = out[0].strip() 
        sm_count = int(108)      # es: 108
        clock_mhz = int(out[1])     # es: 1410
        cc = float(out[2])

    arch_list = architectures.arch_map
    
    specs = arch_list.get(gpu_name, {"bw":1000.0, "cores_sm":64})
    
    # FP32 Peak
    peak_gflops_fp32 = sm_count * specs["cores_sm"] * (clock_mhz / 1000) * 2
    
    # FP64 Peak
    peak_gflops_fp64 = peak_gflops_fp32 / 2

    # Instruction Peak
    peak_gips = sm_count * 4 * (clock_mhz / 1000)

    # Bandwidth Shared Mem
    shared_bw_gbps = sm_count * 32 * 4 * (clock_mhz/1000)

    # Shared Mem Peak
    max_transactions_per_sec = shared_bw_gbps / 32

    bw_l2 = 5373.0
    bw_l1 = 15161.8

    return {
        "gpu_name": gpu_name,
        "peak_fp32": peak_gflops_fp32,
        "peak_fp64": peak_gflops_fp64,
        "bandwidth": specs["bw"],
        "bw_l2": bw_l2,                      
        "bw_l1": bw_l1,
        "sm_count": sm_count,
        "clock_mhz": clock_mhz,
        "peak_gips": peak_gips,              
        "shared_bw": shared_bw_gbps,        
        "trans_bw": max_transactions_per_sec
   }

def fp32_roofline_plot(results, specs):
    print("Generating the FP32 plot")
    
    peak = specs["peak_fp32"]
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
    ax.text(1e1, peak * 1.1, f"Theoretical peak FP32: {peak:.1f} GFLOP/s", ha='center')

    colors = {'L1': '#D49A2A', 'L2': '#2A9D8F', 'HBM': '#6D4C41'}
    markers = {'L1': 'o', 'L2': '^', 'HBM': 's'}

    for mem_level, bw in bw_dict.items():
        color = colors.get(mem_level, 'blue')
        y_vals = x_vals * bw
        
        valid_idx = y_vals <= peak
        ax.plot(x_vals[valid_idx], y_vals[valid_idx], color=color, linewidth=1.2)
        
        ridge_x = peak / bw
        label_x = ridge_x / 3
        label_y = label_x * bw
        ax.text(label_x, label_y * 1.1, f"{mem_level} {bw:.1f} GB/s", 
                color=color, rotation=38, fontsize=10, ha='center', va='bottom')

    perf = results['Performance (GFLOP/s)']

    print(f"  [Debug] FP64 GFLOP/s: {perf:.1f}")
    
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
                ax.scatter(valore_ai, perf, color=color, marker=marker, 
                           s=150, zorder=5, label=f"Kernel ({mem_level})")

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)

    ax.set_xlabel('Arithmetic Intensity (FLOP/B)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title("Hierarchical Roofline Model FP32", fontsize=14)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("roofline_fp32.pdf", format='pdf')
    plt.close()
    print("✅ Plot roofline_fp32.pdf saved") 

def fp64_roofline_plot(results, specs):
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
    ax.text(1e1, peak * 1.1, f"Theoretical peak FP64: {peak:.1f} GFLOP/s", ha='center')

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
        ax.text(label_x, label_y * 1.1, f"{mem_level} {bw:.1f} GB/s", 
                color=color, rotation=38, fontsize=10, ha='center', va='bottom')

    perf = results['Performance (GFLOP/s)']
    
    x_keys_dict = {
        'L1': 'L1 Arithmetic Intensity (FLOP/B)',
        'L2': 'L2 Arithmetic Intensity (FLOP/B)',
        'HBM': 'HBM Arithmetic Intensity (FLOP/B)'
    }

    print(f"  [Debug] FP32 GFLOP/s: {perf:.1f}")
    
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

    ax.set_xlabel('Arithmetic Intensity (FLOP/B)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title("Hierarchical Roofline Model FP64", fontsize=14)
    
    if points_plotted:
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("roofline_fp64.pdf", format='pdf')
    plt.close()
    print("✅ Plot roofline_fp64.pdf saved")


def instruction_roofline_plot(results, specs):

    print("Generating the Instruction Hierarchical plot")
    
    peak_gips = specs.get("peak_gips", 600.0)
    
    bw_dict_bytes = {
        'L1': specs.get("bw_l1", 15161.8),
        'L2': specs.get("bw_l2", 5373.0),
        'HBM': specs.get("bandwidth", 1555.0)
    }
    
    tx_dict = { mem_level: bw / 32.0 for mem_level, bw in bw_dict_bytes.items() }
    
    print(f"  [Debug] Peak GIPS: {peak_gips:.1f}")
    print(f"  [Debug] Transazioni GXTN/s: L1={tx_dict['L1']:.1f}, L2={tx_dict['L2']:.1f}, HBM={tx_dict['HBM']:.1f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(1e0, 1e3)

    ax.hlines(y=peak_gips, xmin=1e-2, xmax=2e3, color='black', linewidth=1.2)
    ax.text(1e1, peak_gips * 1.1, f"Theoretical peak: {peak_gips:.1f} warp GIPS", ha='center')

    colors = {'L1': 'red', 'L2': 'lime', 'HBM': 'blue'}

    for mem_level, tx_rate in tx_dict.items():
        color = colors.get(mem_level, 'black')
        
        ridge_x = peak_gips / tx_rate
        bottom_x = 1e-3 / tx_rate
        
        ax.plot([bottom_x, ridge_x], [1e-3, peak_gips], color=color, linewidth=1.2)
        
        label_x = ridge_x / 3
        label_y = label_x * tx_rate
        ax.text(label_x, label_y * 1.1, f"{mem_level} {tx_rate:.1f} GTXN/s", 
                color=color, rotation=42, fontsize=10, ha='center', va='bottom')

    perf = results.get('Performance GIPS', results.get('GIPS', 0.0))
    valore_ai = results.get('Instruction Intensity', results.get('Instruction_Intensity', 0.0))
    
    print(f"  [Debug] Disegno Kernel a: X={valore_ai:.2f}, Y={perf:.2f}")
    
    points_plotted = False
    if perf > 0 and valore_ai > 0 and not math.isnan(perf) and not math.isnan(valore_ai):
        ax.scatter(valore_ai, perf, color='red', marker='s', 
                   s=150, zorder=5, label="Kernel")
        points_plotted = True

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)

    ax.set_xlabel('Instruction Intensity (warp instructions per transaction)', fontsize=12)
    ax.set_ylabel('Performance (warp GIPS)', fontsize=12)
    
    if points_plotted:
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig("roofline_instructions.pdf", format='pdf')
    plt.close(fig)
    print("✅ Plot roofline_instructions.pdf saved")

def shared_roofline_plot(results, specs):
    print("Generating the Shared Memory plot")
    
    peak_gips = specs.get("peak_gips", 600.0)
    max_tx_rate = peak_gips 
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(1e0, 1e3) 

    
    ax.hlines(y=peak_gips, xmin=1e-2, xmax=2e3, color='black', linewidth=1.2)
    ax.text(1e1, peak_gips * 1.1, f"Theoretical peak: {peak_gips:.1f} warp GIPS", ha='center')

    color_shared = 'magenta'

    bottom_x = 1e-3 / max_tx_rate
    ridge_x = 1.0 
    
    ax.plot([bottom_x, ridge_x], [1e-3, peak_gips], color=color_shared, linewidth=1.2)
    
    label_x = 0.05
    label_y = label_x * max_tx_rate
    ax.text(label_x, label_y * 1.1, f"Shared {max_tx_rate:.1f} GTXA/s", 
            color=color_shared, rotation=42, fontsize=10, ha='center', va='bottom')

    # A) No bank conflict (x = 1.0)
    ax.vlines(x=1.0, ymin=1e0, ymax=peak_gips, color=color_shared, linewidth=1.0)
    ax.text(1.1, 1.5, "No bank conflict", color=color_shared, rotation=90, va='bottom', ha='left', fontsize=10)

    # B) 32-way bank conflict (x = 1/32)
    x_conflict = 1.0 / 32.0
    y_conflict_max = x_conflict * max_tx_rate
    ax.vlines(x=x_conflict, ymin=1e0, ymax=y_conflict_max, color=color_shared, linewidth=1.0)
    ax.text(x_conflict * 1.1, 1.5, "32-way bank conflict", color=color_shared, rotation=90, va='bottom', ha='left', fontsize=10)

    perf = results.get('Performance GIPS Shared', results.get('Performance_GIPS_Shared', 0.0))
    valore_ai = results.get('Shared Intensity', results.get('Shared_Intensity', 0.0))

    print(f"  [Debug] Shared Performance (GIPS): {perf:.2f}")
    print(f"  [Debug] Shared Intensity (warp instr/tx): {valore_ai:.2f}")
    
    points_plotted = False
    
    if perf > 0 and valore_ai > 0 and not math.isnan(perf) and not math.isnan(valore_ai):
        ax.scatter(valore_ai, perf, color=color_shared, marker='s', 
                   s=150, zorder=5, label="Kernel")
        points_plotted = True

    ax.grid(True, which="major", ls="-", color="grey", alpha=0.8)
    ax.grid(True, which="minor", ls="--", color="grey", alpha=0.5)
    ax.set_xlabel('Instruction Intensity (warp instructions per transaction)', fontsize=12)
    ax.set_ylabel('Performance (warp GIPS)', fontsize=12)
    
    if points_plotted:
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig("roofline_shared.pdf", format='pdf')
    plt.close(fig)
    print("✅ Plot roofline_shared.pdf saved")
