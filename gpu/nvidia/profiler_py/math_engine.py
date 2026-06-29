import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def fp32_roofline(df):

    # Calculating the elapsed time in seconds
    df['elapsed_s'] =df['sm__cycles_elapsed.avg'] / df['sm__cycles_elapsed.avg.per_second']

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

    df['Performance_GFLOP_s'] = df['fp32_flops'] / (df['elapsed_s']*1e9)

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

    print(f"The values for fp32 are: {avg_result}")

    return avg_result

def fp64_roofline(df):

    df['fp64_flops'] = ( 
        #  FMA is doubled because it performs both sum and mul
        2 * df['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] +
        df['sm__sass_thread_inst_executed_op_dadd_pred_on.sum']
    )                

    df['Perfomance_GFLOP_s'] = df['fp64_flops'] / (df['elapsed_s']*1e9)

    df['AI_L1'] = df['fp64_flops'] / df['l1tex__t_bytes.sum']
    df['AI_L2'] = df['fp64_flops'] / df['lts__t_bytes.sum']
    df['AI_HBM'] = df['fp64_flops'] / df['dram__bytes.sum']

    avg_result = {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance (GFLOP/s)': df['Performance_GFLOP_s'].mean(),
        'L1 Arithmetic Intensity (FLOP/B)': df['AI_L1'].mean(),
        'L2 Arithmetic Intensity (FLOP/B)': df['AI_L2'].mean(),
        'HBM Arithmetic Intensity (FLOP/B)': df['AI_HBM'].mean()
    }

    return avg_result

def instruction_intensity_roofline(df):
    df['GIPS'] = df['smsp__thread_inst_executed.sum'] / (df['elapsed_s'] * 1e9)

    df['transactions'] = (df['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'] +
                          df['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'])
    
    df['Instruction_Intensity'] = df['smsp__thread_inst_executed.sum'] / df['transactions']

    avg_result = {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance GIPS': df['GIPS'].mean(),
        'Instruction Intensity': df['Instruction_Intensity'].mean()
    }

    return avg_result

def shared_memory_roofline(df):
    # Calcolo Istruzioni Shared
    df['shared_inst'] = (df['smsp__inst_executed_op_shared_ld.sum'] + 
                         df['smsp__inst_executed_op_shared_st.sum'])
    
    # Calcolo Transazioni Shared
    df['shared_transactions'] = (df['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'] + 
                                 df['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'])
    
    df['Performance_GIPS_Shared'] = df['shared_inst'] / (df['elapsed_s'] * 1e9)
    df['Shared_Intensity'] = df['shared_inst'] / df['shared_transactions']

    return {
        'Avg time (s)': df['elapsed_s'].mean(),
        'Performance GIPS Shared': df['Performance_GIPS_Shared'].mean(),
        'Shared Intensity': df['Shared_Intensity'].mean()
    }
