# derp_control
Motion control for the Dual-Rotating Retarder Mueller Polarimeter in the Exopol lab at UC Santa Barbara

Built and tested by William Melby, with some hacking by Jaren Ashcraft

## To run this script

To run the polychromatic mueller polarimetry script, please follow the following steps:

```bash

cd derp_control
conda activate pylablib_env

# run the acquisition script
python measure_polychromatic.py

# run the data reduction script
python load_and_reduce_data.py

```
