import enum

class Mode(enum.Enum):
    # Naive CPU implementation
    NAIVE = 0
    # Naive CPU optimized with Mozart, CPU-only SAs
    MOZART = 1
    # Naive CPU optimized with Bach, accelerator-aware SAs
    BACH = 2
    # Hand-optimized CUDA implementation
    CUDA = 3
