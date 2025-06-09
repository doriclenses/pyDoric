# pyDoric
This repository contains python scripts and [Pyinstaller](https://pyinstaller.org/en/stable/) spec files that are used to create executable files. These executables are distributes as libraries of [danse](https://neuro.doriclenses.com/collections/software/products/danse) software.

## MiniAn
---
To create MiniAn executable, follow the steps: 

1.  Create new environment, and install MiniAn and PyInstaller by running the following in Anaconda prompt:
    ```
    conda create -n minian_pyinstaller -c bioconda -c conda-forge python=3.8 minian h5py pyinstaller pefile=2023.2.7 --yes
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
    conda create -n caiman_pyinstaller -c conda-forge caiman pyinstaller pefile=2023.2.7 ipyparallel=8.8.0 --yes
    ```
    ⚠️ It may take some time to install.

    For more information about the library, please read [CaImAn docs](https://caiman.readthedocs.io/en/latest/Installation.html).

2.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

3.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:

    ```
    pyinstaller /Deploy/pack_caiman_run.spec
    ```

## Suite2p
---

To create Suite2p executable, follow the steps:

1.  Create new environment, and install CaImAn and PyInstaller by running the following in Anaconda prompts:
    1.  ```
        conda create --name suite2p python=3.9 pyinstaller
        ```

    2.  ```
        conda activate suite2p
        ```

    3.  ```
        python -m pip install suite2p
        ```

    For more information about the library, please read [Suite2p docs ](https://suite2p.readthedocs.io/en/latest/installation.html).

2.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

3.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:

    ```
    pyinstaller /Deploy/pack_suite2p_run.spec
    ```

## DeepLabCut
---

1.  [Install CUDA v12.4 or higher](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network)
   
2.  [Install cuDNN v9.1 or higher](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
   
3.  Create new environment, and install MiniAn and PyInstaller by running the following in Anaconda prompt:
    ```
    conda create -n deeplabcut_pyinstaller ??? python=??? ??? pyinstaller ??? --yes
    ```
    For more information about the library, please read [DeepLabCut docs](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/beginners-guide.html).

4.  Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

5.  Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:
    ```
    pyinstaller Deploy/pack_deeplabcut_run.spec
    ```
