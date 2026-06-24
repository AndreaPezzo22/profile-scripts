import subprocess
import pandas as pd 
import io

def profiling_ncu(eseguibile, nome_kernel):
    print(f"\nStarting profiling kernel: '{nome_kernel}'")

    gpu_architecture = query_gpu_architecture()
    metrics_string = build_metrics_string(path_to_metrics, gpu_architecture)

    if not metrics_string:
        print(f"\nNone metric found")
        return None
    
    command = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--kernel-name", nome_kernel,
        "--metrics", metrics_string,
        eseguibile
    ]

    try:
        result = subprocess.run(command, #command that is executed
                                capture_output=True, # stdout and stderr are captured
                                text=True, # file objects for stdin, stdout and stderr are opened in text mode
                                timeout= 10, # Timeout for killing the job
                                check=True) # Arises exceptions
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione di ncu.")
        print(f"Dettagli errore: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Comando ncu non trovato")
        return None
    
    output_ncu = result.stdout

if __name__ == "__main__":
    APP_ESEGUIBILE = "./$HOME/ACA-project/muDock/build/application/muDock" 
    KERNEL_DA_ANALIZZARE = "apply_cuda"
    FILE_METRICHE = "./METRICS"
    
    dati_grezzi_df = profiling_ncu(APP_ESEGUIBILE, KERNEL_DA_ANALIZZARE, FILE_METRICHE)
    