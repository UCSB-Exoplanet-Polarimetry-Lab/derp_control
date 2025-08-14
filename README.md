# derp_control
Motion control and data reduction for the Dual-Rotating Retarder Mueller Polarimeter in the Exopol lab at UC Santa Barbara

Built and tested by William Melby, with some hacking by Jaren Ashcraft.

# Documentation
Documentation for the Derp polarimeter and the associated `derpy` control software can be found in the Github Wiki ([click here](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/derp_control/wiki)). Here you can find installation instructions, a set of tutorials, and a limited parts list.

## Running Derp

To run the polychromatic mueller polarimetry script on Rayleigh, please follow the following steps:

```bash

cd derp_control
conda activate pylablib_env

# run the acquisition script
python measure_polychromatic.py

# run the data reduction script
python load_and_reduce_data.py

```
