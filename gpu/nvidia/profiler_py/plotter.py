import subprocess
import architectures
import numpy as np
import os

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
    
    cmd = ["nvidia-smi", "--query-gpu=name,count.multiprocessors,clocks.max.graphics,compute_cap", "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd).decode("utf-8").strip().split(',')
    
    if len(out) < 4:
        print(f"\nOutput of the arch query gone wrong -> {out}")
        return None
    else:
        gpu_name = out[0].strip() 
        sm_count = int(out[1])      # es: 108
        clock_mhz = int(out[2])     # es: 1410
        cc = float(out[3])

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

    return {
        "gpu_name": gpu_name,
        "peak_fp32": peak_gflops_fp32,
        "peak_fp64": peak_gflops_fp64,
        "bandwidth": specs["bw"],
        "sm_count": sm_count,
        "clock_mhz": clock_mhz,
        "peak_gips": peak_gips,              
        "shared_bw": shared_bw_gbps,        
        "trans_bw": max_transactions_per_sec
    }

def fp32_roofline_plot(results, specs):
    print("Generating the FP32 plot")

    arch_values = specs

    # Theoretical values
    peak = arch_values["peak_fp32"]
    bw = arch_values["bandwidth"]

    plt.figure(figsize=(10, 6))
    plt.xscale('log'); plt.yscale('log')
    
    x_roof = np.logspace(-2, 3, 100)
    y_roof = np.minimum(peak, x_roof * bw)
    plt.plot(x_roof, y_roof, color='black', label="Hardware Limit")

    perf = results['Performance (GFLOP/s)']
    plt.scatter(results['HBM Arithmetic Intensity (FLOP/B)'], perf, color='red', label='HBM')
    plt.xlabel('Arithmetic Intensity (FLOP/B)')
    plt.ylabel('GFLOP/s')
    plt.title('Roofline Model FP32')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig("roofline_fp32.pdf", format='pdf')
    plt.close()
    print("Plot roofline_fp32.pdf saved")

def fp64_roofline_plot(results, specs):
    print("Generating the FP64 plot")

    arch_values = specs

    # Theoretical values
    peak = arch_values["peak_fp64"]
    bw = arch_values["bandwidth"]

    plt.figure(figsize=(10, 6))
    plt.xscale('log'); plt.yscale('log')
    
    x_roof = np.logspace(-2, 3, 100)
    y_roof = np.minimum(peak, x_roof * bw)
    plt.plot(x_roof, y_roof, color='black', label="Hardware Limit")

    perf = results['Performance (GFLOP/s)']
    plt.scatter(results['HBM Arithmetic Intensity (FLOP/B)'], perf, color='red', label='HBM')
    plt.xlabel('Arithmetic Intensity (FLOP/B)')
    plt.ylabel('GFLOP/s')
    plt.title('Roofline Model FP64')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig("roofline_fp64.pdf", format='pdf')
    plt.close()
    print("Plot roofline_fp64.pdf saved")

def instruction_roofline_plot(results, specs):
    print("Generating the Instruction plot")

    arch_values = specs

    # Theoretical values
    peak_gips = arch_values["peak_gips"]
    bw = arch_values["trans_bw"]

    plt.figure(figsize=(10, 6))
    plt.xscale('log'); plt.yscale('log')
    
    x_roof = np.logspace(-2, 3, 100)
    y_roof = np.minimum(peak_gips, x_roof * bw)
    plt.plot(x_roof, y_roof, color='black', label="Hardware Limit")

    perf = results['Performance GIPS']
    plt.scatter(results['Instruction Intensity'], perf, color='red', label='HBM')
    plt.xlabel('Arithmetic Intensity (FLOP/B)')
    plt.ylabel('GFLOP/s')
    plt.title('Roofline Model Instructions')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig("roofline_instructions.pdf", format='pdf')
    plt.close()
    print("Plot roofline_instructions.pdf saved")

def shared_roofline_plot(results, specs):
    print("Generating the Shared Memory plot")

    arch_values = specs

    # Theoretical values
    peak_gips = arch_values["peak_gips"]
    bw = arch_values["trans_bw"]

    plt.figure(figsize=(10, 6))
    plt.xscale('log'); plt.yscale('log')
    
    x_roof = np.logspace(-2, 3, 100)
    y_roof = np.minimum(peak_gips, x_roof * bw)
    plt.plot(x_roof, y_roof, color='black', label="Hardware Limit")

    perf = results['Performance GIPS Shared']
    plt.scatter(results['Shared Intensity'], perf, color='red', label='HBM')
    plt.xlabel('Arithmetic Intensity (FLOP/B)')
    plt.ylabel('GFLOP/s')
    plt.title('Roofline Model Instructions')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig("roofline_shared.pdf", format='pdf')
    plt.close()
    print("Plot roofline_shared.pdf saved")
