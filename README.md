# LoSSETT (Local Scale-to-Scale Energy Transfer Tool)

LoSSETT (the Local Scale-to-Scale Energy Transfer Tool) is a Python package for calculating local energy transfer across specified length scales. The kinetic energy transfer from scales larger than $\ell$ to scales smaller than $\ell$ is derived in [Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312) to be:

$$\mathcal{D}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u} |\delta \mathbf{u}|^2 \mathrm{d}^d \mathbf{r}.$$

Here $\delta \mathbf{u} := \mathbf{u}(\mathbf{x}+\mathbf{r}) - \mathbf{u}(\mathbf{x})$ is a velocity increment, and $G _\ell(\mathbf{r})$ is a filter kernel with characteristic length scale $\ell$. Similar expressions may be derived for the inter-scale transfer of any quadratic scalar (e.g. buoyancy variance, moisture variance).

With thanks to Valerio Lembo (ISAC-CNR) for sharing some of the original MATLAB code developed with colleagues at CNRS ([Kuzzay et al. 2015](https://pubs.aip.org/aip/pof/article-abstract/27/7/075105/103779), [Faranda et al. 2018](https://journals.ametsoc.org/view/journals/atsc/75/7/jas-d-17-0114.1.xml)). Development work funded as part of the UPSCALE project at the University of Reading funded by the Met Office.

Note that this code is currently a work-in-progress.

## Repository Structure

> **Notice:** the functionality in `lossett_control` and `lossett_plotting` will soon only be available in the new repo **[LoSSETT_ANCIL](https://github.com/ElliotMG/LoSSETT_ANCIL)**. This repository will remain focused on the core computation.

---

Within `src`:

* `lossett/calc` contains the core computational routines for calculating inter-scale energy transfers.
* `lossett/filtering` contains utilities for filtering and integration.
* `lossett_control/control` contains Python and Bash scripts for orchestrating the execution of LoSSETT workflows, including on test data.
* `lossett_control/prepreocessing` contains Python scripts for pre-processing data to match the specific cases in `lossett_control/control`.
* `lossett_plotting` contains various example plotting Python scripts and iPython notebooks.

## Prerequisites
Current distribution of python (Python 3) - built on `xarray` with `matplotlib` and `cartopy` for plotting. See `pyproject.toml` for full list of requirements.

## Installation

As a user: activate a suitable environment then pip install:

```bash
pip install  git+https://github.com/ElliotMG/LoSSETT.git
```

As a developer: fork then clone the repository (please create a branch before making any changes!), activate a suitable Python environment, navigate to your LoSSETT directory and

```bash
pip install -e .
```

This will install as the user installation but using the editable cloned code. Please commit code improvements and discuss merging with the master branch with Elliot McKinnon-Gray, Dan Shipley, and other users.

Alternatively:
1. Install poetry [https://github.com/python-poetry/poetry], `pip install poetry`.
2. Clone the repository & install using `poetry`:
   ```bash
   git clone git@github.com:ElliotMG/LoSSETT.git
   cd LoSSETT
   poetry install
   ```
