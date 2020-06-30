# Offload Annotations

This is the main source code repository for Offload Annotations. It contains the Python source code for the Bach runtime, and the benchmarks from the USENIX ATC 2020 paper. The implementation is a prototype -- use at your own risk!

This repository is extended from the [Split Annotations](https://github.com/weld-project/split-annotations) repository from SOSP 2019.

## Example

This simplest example of offload annotations is in `benchmarks/workloads/blackscholes.py`, which uses the [NumPy](https://numpy.org/) library annotated with [PyTorch](https://pytorch.org/) in `pycomposer/sa/annotated/numpy_torch/`. The _annotator_ annotates a CPU library with Offload Annotations:

```
class NdArraySplit(OffloadSplitType):
    def backend(self, value):
        # Whether the value is on the CPU or the GPU.
        if isinstance(value, np.ndarray):
            return Backend.CPU
        elif isinstance(value, torch.Tensor) and value.device.type == 'cuda':
            return Backend.GPU
    def to(self, value, backend):
        # Transfers [value] to specified [device].
        if backend == Backend.GPU:
            return torch.from_numpy(value).to(torch.device('cuda'))
        elif backend == Backend.CPU:
            return value.to(torch.device('cpu')).numpy()
```

The _end user_ imports the annotated library instead of the CPU library,

```
import sa.annotated.numpy_torch as np
```

and explicitly materializes values by calling the provided `evaluate()` function.

```
np.evaluate()
```

## USENIX ATC 2020 Experiments

We ran experiments on a 56-CPU server (2 x Intel E5-2690 v4) with 512GB of memory, running Linux 4.4.0. The machine has a single NVIDIA Tesla P100 GPU with 16GB of memory and CUDA 10.2 installed. The experiments use [Python 3.7](https://www.python.org/downloads/) and the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) package manager to setup a virtual environment.

```
$ conda create -n oas python=3.7 --file environment.yml \
  -c conda-forge -c rapidsai -c nvidia -c pytorch -c numba -c bioconda
$ conda activate oas
```

The entrypoint to any benchmark is `benchmarks/run.py`.

```
$ benchmarks/
$ python run.py --help
usage: Benchmark for offload annotations. [-h] -b BENCHMARK -m MODE [-s SIZE]
                                          [--trials TRIALS]

optional arguments:
  -h, --help            show this help message and exit
  -b BENCHMARK, --benchmark BENCHMARK
                        Benchmark name (blackscholes_torch|blackscholes_cupy|c
                        rime_index|tsvd|pca|dbscan|haversine_torch|haversine_c
                        upy) or (0-7)
  -m MODE, --mode MODE  Mode (cpu|gpu|bach)
  -s SIZE, --size SIZE  Log2 data size
  --trials TRIALS       Number of trials
$ # Run all benchmarks
$ mkdir results/
$ ./run_all.sh
$ ./parse.sh [BENCHMARK_NAME]
```
