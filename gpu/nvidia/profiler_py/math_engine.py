import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

"""
These functionctions compute the real performance values of the application, that are sent to the plotter
to draw the Roofline Models
"""

def fp32_roofline(df):

    # Calculating the elapsed time in seconds
    df['elapsed_s'] =df['sm__cycles_elapsed.avg'] / df['sm__cycles_elapsed.avg.per_second']

    # Summing all the FP32 and FP16 operations (sum, mul)
    df['fp32_flops'] = ( 
        #  FMA is doubled because it performs both sum and mul
        2 * df['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] +
        2 * df['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_hadd_pred_on.sum'] +
	    512 * df["sm__inst_executed_pipe_tensor.sum"]
    )                

    #  Computing the throughput column in GFLOP/s
    df['Performance_GFLOP_s'] = df['fp32_flops'] / (df['elapsed_s']*1e9)

    # Measures how much computational work is perfomed for each transferred byte
    # Allows to understand if the kernel is memory-bound or computationally intensive

    df['AI_L1'] = df['fp32_flops'] / df['l1tex__t_bytes.sum']
    df['AI_L2'] = df['fp32_flops'] / df['lts__t_bytes.sum']
    df['AI_HBM'] = df['fp32_flops'] / df['dram__bytes.sum']

    avg_result = {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance (GFLOP/s)': df['Performance_GFLOP_s'].mean(),
        'L1 Arithmetic Intensity (FLOP/B)': df['AI_L1'].mean(),
        'L2 Arithmetic Intensity (FLOP/B)': df['AI_L2'].mean(),
        'HBM Arithmetic Intensity (FLOP/B)': df['AI_HBM'].mean()
    }

    print(f"\nThe avg time is: {avg_result['Avg time (s)']:.4f}")

    return avg_result

def fp64_roofline(df):

    # Summing all the FP64 operations
    df['fp64_flops'] = ( 
        #  FMA is doubled because it performs both sum and mul
        2 * df['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_dadd_pred_on.sum']
    )                

    # Computing the throughput in GFLOP/s
    df['Performance_GFLOP_s_FP64'] = df['fp64_flops'] / (df['elapsed_s']*1e9)

    df['AI_L1'] = df['fp64_flops'] / df['l1tex__t_bytes.sum']
    df['AI_L2'] = df['fp64_flops'] / df['lts__t_bytes.sum']
    df['AI_HBM'] = df['fp64_flops'] / df['dram__bytes.sum']

    avg_result = {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance FP64 (GFLOP/s)': df['Performance_GFLOP_s_FP64'].mean(),
        'L1 Arithmetic Intensity (FLOP/B)': df['AI_L1'].mean(),
        'L2 Arithmetic Intensity (FLOP/B)': df['AI_L2'].mean(),
        'HBM Arithmetic Intensity (FLOP/B)': df['AI_HBM'].mean()
    }

    return avg_result

def instruction_intensity_roofline(df):

    # Computes the throughput of all the instructions executed per second (GIPS)
    df['GIPS'] = df['smsp__thread_inst_executed.sum'] / (df['elapsed_s'] * 1e9)

    # Sums the load and store global transactions
    df['transactions_l1'] = (df['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'] +
                          df['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'])
    
    df['transactions_l2'] = (df['lts__t_sectors_op_read.sum'] +
                          df['lts__t_sectors_op_write.sum'])
    
    df['transactions_hbm'] = (df['dram__sectors_read.sum'] +
                          df['dram__sectors_write.sum'])
    

    
    # Divides the number of instructions by the number of transactions, measuring how many instructions are
    # executed for each memory transaction
    df['Instruction_Intensity_L1'] = df['smsp__thread_inst_executed.sum'] / df['transactions_l1']
    df['Instruction_Intensity_L2'] = df['smsp__thread_inst_executed.sum'] / df['transactions_l2']
    df['Instruction_Intensity_HBM'] = df['smsp__thread_inst_executed.sum'] / df['transactions_hbm']

    avg_result = {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance GIPS': df['GIPS'].mean(),
        'Instruction Intensity L1': df['Instruction_Intensity_L1'].mean(),
        'Instruction Intensity L2': df['Instruction_Intensity_L2'].mean(),
        'Instruction Intensity HBM': df['Instruction_Intensity_HBM'].mean()
    }

    return avg_result

def shared_memory_roofline(df):
    # Summing the shared memory operations
    df['shared_inst'] = (df['smsp__inst_executed_op_shared_ld.sum'] + 
                         df['smsp__inst_executed_op_shared_st.sum'])
    
    # Computing the shared memory transactions
    df['shared_transactions'] = (df['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'] + 
                                 df['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'])
    
    # Throughput Shared operations
    df['Performance_GIPS_Shared'] = df['shared_inst'] / (df['elapsed_s'] * 1e9)

    # Measuring how many instructions are executed for each shared memory transaction
    df['Shared_Intensity'] = df['shared_inst'] / df['shared_transactions']

    return {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance GIPS Shared': df['Performance_GIPS_Shared'].mean(),
        'Shared Intensity': df['Shared_Intensity'].mean()
    }
