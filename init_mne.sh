#!/bin/bash

# Reset modules and load bear core modules
module purge; module load bluebear
module load bear-apps/2021a/live
# Load core python version
module load Python/3.9.5-GCCcore-10.3.0
# Load pre-compiled python modules, the versions must match either 
# the year of the bear-apps release (eg '2021a' from bear-apps/2021a/live) or 
# the GCCcore version used in the main python module (eg 'GCCcore-10.3.0' from Python/3.9.5-GCCcore-10.3.0)
# any mismatch in either will cause a problem...
module load IPython/7.25.0-GCCcore-10.3.0
module load numba/0.53.1-foss-2021a
module load Tkinter/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a
module load scikit-learn/0.24.2-foss-2021a

# Set some path names
export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/mne-${BB_CPU}"

# Create master dir if necessary
mkdir -p ${VENV_DIR}
echo ${VENV_PATH}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
	python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate virtual environment
source ${VENV_PATH}/bin/activate

# Any additional installations - can use pip as normal here. 
# Generally preferred to load a module in the earlier section if possible though.
pip install --upgrade pip
pip install scikit-learn
pip install mne
pip install rsatoolbox
pip install opencv-python
pip install glmtools
pip install tabulate

# pip install spyder  # Got some errors when including this.
