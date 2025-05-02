# LoSSETT (Local Scale-to-Scale Energy Transfer Tool)

LoSSETT (the Local Scale-to-Scale Energy Transfer Tool) is a program designed to calculate local energy transfer across specified length scales. The energy transfer from scales larger than $\ell$ to scales smaller than $\ell$ is derived in [Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312) to be:

$\mathbf{D}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u} |\delta \mathbf{u}|^2 \mathrm{d}^d\mathbf{r}.$

With thanks to Valerio Lembo (ISAC-CNR) for sharing some of the original MATLAB code developed with colleagues at CNRS ([Kuzzay et al. 2015](https://pubs.aip.org/aip/pof/article-abstract/27/7/075105/103779), [Faranda et al. 2018](https://journals.ametsoc.org/view/journals/atsc/75/7/jas-d-17-0114.1.xml)). Development work funded as part of the UPSCALE project at the University of Reading funded by the Met Office.

Note that this code is currently a work-in-progress.

## Repository Structure

* `/calc/` contains the core computational routines for calculating inter-scale energy transfers and related metrics:
* `/control/` contains scripts for orchestrating the execution of LoSSETT workflows, including on test data.
* `/filtering/` contains utilities for filtering and integration.
* `/plotting/` contains various example plotting `.py` scripts and `.ipynb` notebooks.
* `/preprocessing/` contains python scripts for pre-procesing the data to match the specific cases in `/control`.

## Prerequisites
Current distribution of python (Python 3) - built on `xarray` with `matplotlib` and `cartopy` for plotting. See `requirements.txt` for full list of requirements.

## Installation

1. Clone the repository:
   ```bash
   git clone github.com/ElliotMG/LoSSETT/
   cd LoSSETT
