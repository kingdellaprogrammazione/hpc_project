#!/bin/bash

CPUS=$1
MATRIX_SIDE=$2
NODES=2                  # or compute from your environment / parameters

TPN=$(( CPUS / NODES ))

sbatch \
  --ntasks=$CPUS \
  --nodes=$NODES \
  --ntasks-per-node=$TPN \
  launcher_test_v2.sh $MATRIX_SIDE