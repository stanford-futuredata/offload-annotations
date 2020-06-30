import enum

class Mode(enum.Enum):
    # Naive CPU implementation
    CPU = 0
    # Naive CPU optimized with Mozart, CPU-only SAs
    MOZART = 1
    # Naive CPU optimized with Bach, accelerator-aware OAs
    BACH = 2
    # Hand-optimized GPU implementation
    GPU = 3
