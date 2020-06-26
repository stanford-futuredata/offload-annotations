## Offload Annotations

This is the main source code repository for Offload Annotations. It contains the Python source code for the Bach runtime, and the benchmarks from the USENIX ATC 2020 paper.

## Tests

```
pytest benchmarks/test.py -m "not bach"
pytest benchmarks/test.py -k TestHaversine
```