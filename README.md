# Offload Annotations

This is the main source code repository for Offload Annotations. It contains the Python source code for the Bach runtime, and the benchmarks from the USENIX ATC 2020 paper.

## USENIX ATC 2020 Experiments

This section describes how to run the main experiments in the USENIX ATC 2020 paper.

### System Requirements

We ran experiments on a 56-CPU server (2 x Intel E5-2690 v4) with 512GB of memory, running Linux 4.4.0. The machine has a single NVIDIA Tesla P100 GPU with 16GB of memory and CUDA 10.2 installed.

### Experiments

The experiments require Python 3.6 and the Conda package manager. Setup a virtual environment.

```
conda create -n oas python=3.6 --file requirements.txt \
  -c conda-forge -c rapidsai -c nvidia -c pytorch -c numba -c bioconda
conda activate oas
```

Run all experiments.

```
cd benchmarks/
./run_all.sh
./parse.sh [BENCHMARK_NAME]
```

## Tests

```
cd benchmarks/
pytest test.py -m "not bach"
pytest test.py -k TestHaversine
```
