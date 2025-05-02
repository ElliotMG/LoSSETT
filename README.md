# LoSSETT (Local Scale-to-Scale Energy Transfer Tool)

LoSSETT (the Local Scale-to-Scale Energy Transfer Tool) is a program designed to calculate local energy transfer across specified length scales using the methodology derived by [Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312).

With thanks to Valerio Lembo (ISAC-CNR) for sharing some of the original MATLAB code developed with colleagues at CNRS ([Kuzzay et al. 2015](https://pubs.aip.org/aip/pof/article-abstract/27/7/075105/103779), [Faranda et al. 2018](https://journals.ametsoc.org/view/journals/atsc/75/7/jas-d-17-0114.1.xml)).

## Repository Structure

### `/calc/`
This directory contains the core computational routines for calculating inter-scale energy transfers and related metrics:
- **`calc_inter_scale_transfers.py`**: Implements the main algorithms for computing $\mathcal{D}_\ell$.

### `/control/`
Scripts for orchestrating the execution of LoSSETT workflows:
- **`run_lossett_era5_0p5deg.py`**: Runs LoSSETT configured for ERA5 data.
- **`run_lossett_kscale_0p5deg.py`**: Runs LoSSETT configured for Met Office K-Scale data (e.g. see [Jones et al 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104672)).
- **`run_lossett_u-dc009.py`**: Runs LoSSETT for a long-term 10km simulation of the North Atlantic to examine hurricane energetic structures.
- **`submit_calc_DR_indicator_DS_CTC5RAL.sh`**: Example SLURM script for submitting batch jobs to compute $\mathcal{D}_\ell$.

### `/filtering/`
This directory contains utilities for filtering and integration:
- **`get_integration_kernels.py`**: Calculates filter kernels $G_\ell (r)$ defined in **`kernels.py`**.

### `/plotting/`
Contains various example plotting `.py` scripts and `.ipynb` notebooks.

### `/preprocessing/`
Python scripts for pre-procesing the data to match the specific cases in **`/control`**.

## Prerequisites
Current distribution of python (Python 3).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LoSSETT
