#!/bin/bash
set -e

function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${DIR}/../../..
source $BASE_DIR/venv/bin/activate
export PYTHONPATH=${BASE_DIR}/venv/bin/python/dist-packages
echo "Entered virtualenv."


# A B C D E D C B A
