# TO DO

## Installing Miniconda Environment

### Install the Environment
1. 
   ```bash
   mkdir -p ~/miniconda3
   ```
2. 
   ```bash
   wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O ~/miniconda3/miniconda.sh
   ```
3. 
   ```bash
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   ```
4. 
   ```bash
   ~/miniconda3/bin/conda init bash
   ```
5. 
   ```bash
   source ~/.bashrc
   ```

### Accept the conditions
6. 
   ```bash
   conda tos accept --override-channels --channel [https://repo.anaconda.com/pkgs/main](https://repo.anaconda.com/pkgs/main)
   ```
7. 
   ```bash
   conda tos accept --override-channels --channel [https://repo.anaconda.com/pkgs/r](https://repo.anaconda.com/pkgs/r)
   ```

### Creating the New Conda Environment
The file `environment.yml` contains all the needed dependencies to run the profiler on CUDA kernels, if you need to install other packets.

8. 
   ```bash
   conda env create -f environment.yml -n demo_env
   ```
9. 
   ```bash
   conda activate demo_env
   ```

## Clone and Compile Application to Profile

### Clone the muDock Application
1. Create a new directory: 
   ```bash
   mkdir -p ~/ACA-project
   ```
2. Move inside the new dir: 
   ```bash
   cd ~/ACA-project
   ```
3. Clone the muDock application: 
   ```bash
   git clone [https://github.com/elvispolimi/muDock.git](https://github.com/elvispolimi/muDock.git)
   ```

### Clone the Profiler
4. 
   ```bash
   git clone [https://github.com/elvispolimi/profile-scripts.git](https://github.com/elvispolimi/profile-scripts.git)
   ```

### Compiler muDock application
In this case we are profiling muDock application, so the following instructions are ad-hoc. If you have already compiled your personal application skip to the next step.

5. 
   ```bash
   cd ~/ACA-project/muDock
   ```
6. 
   ```bash
   export CMAKE_PREFIX_PATH=$CONDA_PREFIX
   ```
7. 
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMUDOCK_ENABLE_CUDA=ON -DMUDOCK_GPU_ARCHITECTURES=cuda:sm_80
   ```
8. 
   ```bash
   cmake --build build
   ```

### Fix the Lock File
9. 
   ```bash
   mkdir -p ~/ACA-project/ncu_tmp
   ```
10. 
    ```bash
    export TMPDIR=~/ACA-project/ncu_tmp
    ```

## Run the Profiler

### Executing the Profiler
Execute the profiling of `calc_energy` kernel, with `1fkb_pocket.pdb` and as ligand `mixed_50k.adtmol2`
```bash
python3 main.py -k calc_energy -e ~/ACA-project/muDock/build/application/muDock -wd ~/ACA-project/muDock/data/1fkb/ -- --protein ~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb --ligand ~/ACA-project/muDock/data/mixed_50k.adtmol2 --use CUDA:GPU:0
```

Execute the profiler of `calc_energy` kernel, with `1fkb_pocket.pdb` and as ligand `1fkb_ligand.mol2`
```bash
python3 main.py -k calc_energy -e ~/ACA-project/muDock/build/application/muDock -wd ~/ACA-project/muDock/data/1fkb/ -- --protein ~/ACA-project/muDock/data/1fkb/1fkb_pocket.pdb --ligand ~/ACA-project/muDock/data/1fkb/1fkb