#!/bin/bash

CPUS=$1
MATRIX_SIDE=$2


sbatch \
  --ntasks=$CPUS \
  launcher_test_v2.sh $MATRIX_SIDE