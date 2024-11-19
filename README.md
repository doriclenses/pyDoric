# pyDoric
This repository contains python scripts and [Pyinstaller](https://pyinstaller.org/en/stable/) spec files that are used to create executable files. These executables are distributes as libraries of [danse](https://neuro.doriclenses.com/collections/software/products/danse) software.

## MiniAn
---

[Install MiniAn](https://minian.readthedocs.io/en/stable/start_guide/install.html) and Pyinstaller in the same anaconda environment.

Start a command prompt in Minian environment and go to the git directory.

```
cd PATHTOGITFOLDER
```

run the command pyinstaller

```
pyinstaller pack_minian_run.spec
```

## Caiman
---

[Install CaImAn](https://caiman.readthedocs.io/en/latest/Installation.html) and Pyinstaller in the same anaconda environment and set up caimanmanager to get **caiman_data** folder. ⚠️Then copy **caiman_data** folder into the folder where Caiman will be compiled (git folder)

Start a command prompt in Caiman environment and go to the git directory.

```
cd PATHTOGITFOLDER
```

then run pyinstaller

```
pyinstaller pack_caiman_run.spec
```
