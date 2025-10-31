# pyDoric
This repository contains python scripts and [Pyinstaller](https://pyinstaller.org/en/stable/) spec files that are used to create executable files. These executables are distributes as libraries of [danse](https://neuro.doriclenses.com/collections/software/products/danse) software.

## MiniAn
---
To create MiniAn executable, follow the steps: 

1.  Create new environment, and install MiniAn and PyInstaller by running the following in Anaconda prompt:
    ```
    conda create -n minian_pyinstaller -c bioconda -c conda-forge python=3.8 minian=1.2.1 h5py pyinstaller pefile=2023.2.7 --yes
    ```
    ```
    conda activate minian_pyinstaller
    ```
    ⚠️ It may take some time to install.    

    For more information about the library, please read [MiniAn docs](https://minian.readthedocs.io/en/stable/start_guide/install.html).

2.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

3.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:

    ```
    pyinstaller /Deploy/pack_minian_run.spec
    ```

## CaImAn
---
To create CaImAn executable, follow the steps:

1.  Create new environment, and install CaImAn and PyInstaller by running the following in Anaconda prompt:
    ```
    conda create -n caiman_pyinstaller -c conda-forge python=3.10.8 caiman=1.11.4 pyinstaller pefile=2023.2.7 ipyparallel=8.8.0 --yes
    ```
    ```
    conda activate caiman_pyinstaller
    ```
    ⚠️ It may take some time to install.
   
    For more information about the library, please read [CaImAn docs](https://caiman.readthedocs.io/en/latest/Installation.html).

2.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

3.  Use CaImAn manager to finish the istallation by running in Anaconda prompt:
    ```
    caimanmanager install
    ```
    This would add 'caiman_data' folder into your user folder 'C:\Users\your_user_name\'. Copy this this folder into your cloned 'CaImAn' folder. 


4.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:
    ```
    pyinstaller /Deploy/pack_caiman_run.spec
    ```

## Suite2p
---

To create Suite2p executable, follow the steps:

1.  Create new environment, and install CaImAn and PyInstaller by running the following lines in Anaconda prompt:
    ```
    conda create --name suite2p_pyinstaller python=3.9 pyinstaller h5py --yes
    ```
    ```
    conda activate suite2p_pyinstaller
    ```
    ```
    python -m pip install suite2p
    ```
    For more information about the library, please read [Suite2p docs](https://suite2p.readthedocs.io/en/latest/installation.html).

3.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

4.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:

    ```
    pyinstaller /Deploy/pack_suite2p_run.spec
    ```

## DeepLabCut
---

1.  [Install CUDA v12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows)
   
2.  [Install cuDNN v9.6](https://developer.nvidia.com/cudnn-9-6-0-download-archive?target_os=Windows)
   
3.  Create new environment, and install DeepLabCut and PyInstaller by running the following in Anaconda prompt:
    ```
    conda create -n deeplabcut_pyinstaller -c conda-forge python=3.10 pytables==3.8.0 pyinstaller h5py --yes
    ```
    ```
    conda activate deeplabcut_pyinstaller
    ```
    ```
    pip install torch torchvision
    ```
    ```
    pip install "deeplabcut[gui,modelzoo,wandb]==3.0.0rc8"
    ```

    For more information about the library, please read [DeepLabCut docs](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/beginners-guide.html).

4.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

5.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:
    ```
    pyinstaller Deploy/pack_deeplabcut_run.spec
    ```
