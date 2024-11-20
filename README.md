# pyDoric
This repository contains python scripts and [Pyinstaller](https://pyinstaller.org/en/stable/) spec files that are used to create executable files. These executables are distributes as libraries of [danse](https://neuro.doriclenses.com/collections/software/products/danse) software.

## MiniAn
---

[Install MiniAn](https://minian.readthedocs.io/en/stable/start_guide/install.html) and Pyinstaller in the same anaconda environment.

Start a command prompt in Minian environment and go to the git directory.

```
cd PATHTOGITFOLDER/Deploy
```

run the command pyinstaller

```
pyinstaller pack_minian_run.spec
```

## CaImAn
---

[Install CaImAn](https://caiman.readthedocs.io/en/latest/Installation.html) and Pyinstaller in the same anaconda environment and set up caimanmanager to get **caiman_data** folder. ⚠️Then copy **caiman_data** folder into the folder where CaImAn will be compiled (git folder)

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

Start a command prompt in Caiman environment and go to the git directory.

```
cd PATHTOGITFOLDER/Deploy
```

then run pyinstaller

```
pyinstaller pack_suite2p_run.spec
```
