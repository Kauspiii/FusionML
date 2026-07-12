import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

import mlx.core as mx
from fusionml._metal.tri_scheduler import get_scheduler

scheduler = get_scheduler()
shapes = [
    (1024, 1600, 1600),
    (1024, 1600, 4800),
    (1024, 1600, 6400),
    (1024, 6400, 1600),
    (1024, 1600, 1024),
    (1024, 1024, 1600),
    (1600, 1600, 1024),
    (6400, 1600, 1024),
    (1600, 6400, 1024)
]
scheduler.calibrate(shapes=shapes, verbose=True)
scheduler.print_status()
