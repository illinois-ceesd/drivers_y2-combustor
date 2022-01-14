#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py combustor.py -i run_params.yaml -r restart_data/combustor-000010 --log
