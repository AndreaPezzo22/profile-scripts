import subprocess
import pandas as pd
import io
import os

def detect_architecture():
    """
    Interroga i driver NVIDIA per scoprire automaticamente l'architettura della GPU.
    Restituisce un numero intero (es. 80 per Ampere, 89 per Ada, 90 per Hopper).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        cc_stringa = result.stdout.strip().split('\n')[0]
        architecture = int(float(cc_stringa) * 10)
        print(f"Architettura GPU rilevata automaticamente: {architecture}")
        return architecture
    except Exception as e:
        print(f"Impossibile rilevare l'architettura GPU ({e}). Uso fallback: 80")
        return 80

def build_metrics_string(path_to_metrics, architecture):
    """
    Legge il file METRICS, rimuove i commenti con .split('#'), scarta le metriche L1 
    hardcodate e inserisce quelle dinamiche corrette per l'architettura della GPU.
    """
    if not os.path.exists(path_to_metrics):
        print(f"File delle metriche non trovato: {path_to_metrics}")
        return ""

    # Metriche della Cache L1 da escludere per evitare crash di incompatibilità hardware
    l1_da_escludere = [
        "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained",
        "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained",
        "l1tex__lsu_writeback_active.sum.peak_sustained",
        "l1tex__lsu_writeback_active_mem_lgds.sum.per_second",
        "l1tex__lsu_writeback_active_mem_lg.sum.per_second",
        "l1tex__lsu_writeback_active.sum.per_second"
    ]

    metriche_valide = []
    
    with open(path_to_metrics, 'r') as f:
        for each_line in f:
            # Corretto: uso .split('#') per tagliare i commenti
            metric = each_line.split('#')[0].strip()
            if metric and metric not in l1_da_escludere:
                metriche_valide.append(metric)

    # Inserimento dinamico delle metriche L1 in base all'architettura
    if architecture >= 90:
        l1_peak = "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active_mem_lgds.sum.per_second"
    elif architecture in [75, 80, 86, 87, 88, 89]:
        l1_peak = "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active_mem_lg.sum.per_second"
    else:
        l1_peak = "l1tex__lsu_writeback_active.sum.peak_sustained"
        l1_per_sec = "l1tex__lsu_writeback_active.sum.per_second"

    metriche_valide.extend([l1_peak, l1_per_sec])
    
    print(f"Caricate con successo {len(metriche_valide)} metriche hardware dal file.")
    return ",".join(metriche_valide)

def profiling_ncu(eseguibile, argomenti_app, nome_kernel, path_to_metrics, cartella_lavoro):
    """
    Lancia Nsight Compute, passa gli argomenti a muDock ed estrae il CSV grezzo.
    """
    print(f"\nAvvio profiling per il kernel: '{nome_kernel}'...")

    gpu_architecture = detect_architecture()
    metrics_string = build_metrics_string(path_to_metrics, gpu_architecture)    
    if not metrics_string:
        print("\nNessuna metrica valida trovata.")
        return None

    output_file = "ncu_report.csv"

    command = [
        "ncu",
        "--csv",
        "--page", "raw",
        "--kernel-name", nome_kernel,
        "--metrics", metrics_string,
        "--export", output_file,
        eseguibile
    ]
    command.extend(argomenti_app)
    
    print(f"The full command is : {command}")

    try:
        # Timeout impostato a 20 minuti (1200 secondi) per evitare blocchi parziali
        # result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1200, cwd=cartella_lavoro)
        subprocess.run(command, cwd=cartella_lavoro)
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione di ncu.")
        print(f"Dettagli errore: {e.stderr}")
        return None
    except FileNotFoundError:
        print("❌ Comando 'ncu' non trovato. Assicurati che Nsight Compute sia nel PATH.")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Timeout! L'applicazione ha impiegato troppo tempo.")
        return None

    percorso_csv_completo = os.path.join(cartella_lavoro if cartella_lavoro else ".", output_file)

    if os.path.exists(output_file + ".csv"):
        df = pd.read_csv(output_file + ".csv")
        print("Dati caricati dal file")  
        return df
    else:
        print("File csv non creato")
        return None


if __name__ == "__main__":
    # 1. Configurazione Percorsi
    APP_ESEGUIBILE = os.path.expanduser("~/ACA-project/muDock/build/application/muDock")
    FILE_METRICHE = os.path.expanduser("~/ACA-project/profile-scripts/gpu/nvidia/METRICS")
    
    percorso_proteina = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb")
    percorso_ligando = os.path.expanduser("~/ACA-project/muDock/data/1fkb/1fkb_ligand.mol2")
    CARTELLA_DI_LAVORO = os.path.expanduser("~/ACA-project/muDock/data/1fkb/")

    # 2. Configurazione Profiling
    KERNEL_DA_ANALIZZARE = "apply_cuda"
    ARGOMENTI_MUDOCK = [
        "--protein", percorso_proteina,
        "--ligand", percorso_ligando,
        "--use", "CUDA:GPU:0"
    ]

    # 3. Avvio della pipeline
    dati_grezzi_df = profiling_ncu(
        eseguibile=APP_ESEGUIBILE, 
        argomenti_app=ARGOMENTI_MUDOCK, 
        nome_kernel=KERNEL_DA_ANALIZZARE, 
        path_to_metrics=FILE_METRICHE,
        cartella_lavoro=CARTELLA_DI_LAVORO
    )
