"""Offline CARLA training and evaluation for the STA controller."""

import os

# Workers own their MuJoCo state; avoid nested BLAS thread pools in each worker.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
del _name
