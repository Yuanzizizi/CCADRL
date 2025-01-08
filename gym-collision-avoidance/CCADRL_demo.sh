#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/gym_collision_avoidance/experiments/utils.sh

print_header "Running CCADRL python script"

cd $DIR
python src/example.py --checkpt_name "ccadrl"


# A B C D E D C B A
