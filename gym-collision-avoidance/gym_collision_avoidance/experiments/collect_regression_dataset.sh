#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Collecting Trajectory Dataset"

# # Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Experiment
cd $DIR
python src/collect_regression_dataset.py


# A B C D E D C B A
