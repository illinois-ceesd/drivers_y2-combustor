#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py combustor.py -i run_params.yaml --log --lazy 2>>stderr.out 1>>stdout.out
