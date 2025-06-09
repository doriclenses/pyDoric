# pyDoric
This repository contains python scripts and [Pyinstaller](https://pyinstaller.org/en/stable/) spec files that are used to create executable files. These executables are distributes as libraries of [danse](https://neuro.doriclenses.com/collections/software/products/danse) software.

## MiniAn
---
To create MiniAn executable, follow the steps: 

1. Create new environment, and install MiniAn and PyInstaller by running the following in Anaconda prompt:
```
conda create -n minian_pyinstaller -c bioconda -c conda-forge python=3.8 minian h5py pyinstaller pefile=2023.2.7 --yes
```
⚠️ It may take some time to install.

For more information about the library, please read [MiniAn docs](https://minian.readthedocs.io/en/stable/start_guide/install.html).

2. Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

3. Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:

```
pyinstaller /Deploy/pack_minian_run.spec
```

## CaImAn
---

[Install CaImAn](https://caiman.readthedocs.io/en/latest/Installation.html) and Pyinstaller in the same anaconda environment and set up caimanmanager to get **caiman_data** folder. ⚠️Then copy **caiman_data** folder into the folder where CaImAn will be compiled (git folder)

### To install CaImAn

To install CaImAn fallow the fallowing steps.  ⚠️ It could take some time to install.

Open a command prompt in Anaconda then type 

```
conda create -n caiman_pyinstaller -c conda-forge caiman pyinstaller pefile=2023.2.7 ipyparallel=8.8.0 --yes
```

### To Package CaImAn using pyinstaller
Start a command prompt in CaImAn environment and go to the git directory.

```
cd PATHTOGITFOLDER/Deploy
```

then run pyinstaller

```
pyinstaller pack_caiman_run.spec
```

## Suite2p
---

[Install Suite2p](https://suite2p.readthedocs.io/en/latest/installation.html) and Pyinstaller in the same anaconda environment.

### To install Suite2p
To install Suite2p fallow the fallowing steps.  ⚠️ It could take some time to install.

Open a command prompt in Anaconda then type
```
conda create --name suite2p python=3.9
```

Then activate the newly created environement
```
conda activate suite2p
```

Then install suite2p 
```
python -m pip install suite2p
```

### To Package Suite2p using pyinstaller
Start a command prompt in Suite2p environment and go to the git directory.

```
cd PATHTOGITFOLDER/Deploy
```

then run pyinstaller

```
pyinstaller pack_suite2p_run.spec
```

## DeepLabCut
---

1. [Install CUDA v12.4 or higher](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network)
   
2. [Install cuDNN v9.1 or higher](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
   
3. Create new environment, and install MiniAn and PyInstaller by running the following in Anaconda prompt:
```
conda create -n deeplabcut_pyinstaller ??? python=??? ??? pyinstaller ??? --yes
```
For more information about the library, please read [DeepLabCut docs](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/beginners-guide.html).

4. Clone this repository. In Anaconda prompt, navigate to the directory where the repository was cloned.

5. Package the code into an executable by using pyinstaller spec file and running the following in Anaconda prompt:
```
pyinstaller Deploy/pack_deeplabcut_run.spec
```
