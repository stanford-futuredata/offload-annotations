from .backend import Backend

config = {
        # The maximum number of workers spawned.
        "workers": 2,
        # The default batch size for each backend when one cannot be discovered automatically.
        "batch_size": {
            Backend.CPU: 16384,
            Backend.GPU: 524288,
        },
    }
