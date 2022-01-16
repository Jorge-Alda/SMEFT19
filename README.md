# SMEFT19



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5781796.svg)](https://doi.org/10.5281/zenodo.5781796)


In this repository you can find the code that I have used to investigate the impact of New Physics in the form of the
SMEFT operators *Olq(1)* and *Olq(3)*, with various flavour structures, to a large range of physical observables.

* [On-line documentation](https://jorge-alda.github.io/SMEFT19)
* [PDF documentation](https://raw.githubusercontent.com/Jorge-Alda/SMEFT19/gh-pages/SMEFT19.pdf)
* [Jupyter notebooks](https://jorge-alda.github.io/SMEFT19-notebooks/index.html)
* [Source code for the Jupyter notebooks](https://github.com/Jorge-Alda/SMEFT19-notebooks)

This code was used for the following papers:
* Jorge Alda, Jaume Guasch, Siannah Penaranda, *Anomalies in B mesons decays: A phenomenological approach*. [![](https://img.shields.io/badge/arXiv-2012.14799-00ff00)](https://arxiv.org/abs/2012.14799)
* Jorge Alda, Jaume Guasch, Siannah Penaranda, *Anomalies in B mesons decays: Present status and future collider prospects*. Contribution to [LCWS2021](https://www.slac.stanford.edu/econf/C2103151/). [![](https://img.shields.io/badge/arXiv-2105.05095-00ff00)](https://arxiv.org/abs/2105.05095)
* Jorge Alda, Jaume Guasch, Siannah Penaranda, *Using Machine Learning techniques in phenomenological studies in flavour physics*. [![](https://img.shields.io/badge/arXiv-2109.07405-00ff00)](https://arxiv.org/abs/2109.07405)
* Jorge Alda, Jaume Guasch, Siannah Penaranda, *Exploring B-physics anomalies at colliders*. Contribution to [EPS-HEP2021](https://www.eps-hep2021.eu/). [![](https://img.shields.io/badge/arXiv-2110.12240-00ff00)](https://arxiv.org/abs/2110.12240)

## How to run this code

The recommended way to run `SMEFT19` is using the [Docker](https://docs.docker.com/engine/install/) containers. In this way, you will have the right dependencies without having to worry about conflicts with the rest of your system. First, download the [container](https://hub.docker.com/repository/docker/jorgealda/smeft19) (it may take some time, only needed for the first time) using the command

```bash
docker pull jorgealda/smeft19
```

and launch it in interactive mode

```bash
docker run -it jorgealda/smeft19
```

This will start an ubuntu shell, meaning that you have the
basic Unix tools (`less`, `ls`, `cp`, `sed`,...) at your disposal. There is also a `python3.8` interpreter installed, that you can start by typing

```bash
python3
```

Python has all the `SMEFT19` dependecies already installed:
`numpy`, `scipy`, `pandas`, `matplotlib` (but there is no graphic display), `wilson`, `flavio`, `smelli`, `sklearn`, `xgboost` and `shap`. 

Let's see a simple example which loads `SMEFT19`, computes the prediction for RK+ in scenario IV and writes it to a file. Enter the following in the `Python3` interpreter:

```python
from SMEFT19.SMEFTglob import prediction
from SMEFT19.scenarios import scIV
pred = prediction([-0.2, 0.21], ('<Rmue>(B+->Kll)', 1.1, 6.0), scIV)
with open('prediction.txt', 'wt') as f:
    f.write(f'RK+ = {pred}')
exit()
```

To see the contents of the file that you created, execute

```bash
less prediction.txt
```

and you should get the following: `RK+ = 0.8175725039919868`

For now, we can stop the container simply typing

```bash
exit
```

If you want to retrieve a file saved in the container, you first need to know the ID of the container that created it, by running the command 

```bash
docker ps -a | grep smeft19
```

This will produce a list of all `smeft19` currently running or stopped:

```txt
b11cd92478b9   jorgealda/smeft19                              "bash"    9 minutes ago   Exited (0) 2 minutes ago             recursing_rubin
```

Once you have localized the right container (if you have more than one, check the start and exit times), its ID will be the hexadecimal string at the begining of the line, in this case `b11cd92478b9` (yours will be different). 

You can copy the file from the container to your system using Docker's `cp`:

```bash
docker cp b11cd92478b9:/prediction.txt ./prediction.txt
```

You can return to a stopped container using

```bash
docker start b11cd92478b9 && docker attach b11cd92478b9
```

And finally you can delete a container that you no longer need (you will loose all files in the container that are not `cp`'ed to your system) with the command

```bash
docker rm b11cd92478b9
```
