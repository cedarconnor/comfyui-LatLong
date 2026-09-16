"""Manual benchmark: python tests/benchmark.py 2048 4096 16384."""
import json
import os
import sys
import time
import threading
import platform
import psutil
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from run_node_smoke_tests import stub_comfy_modules, load_latlong_nodes_package

stub_comfy_modules()
P = load_latlong_nodes_package(os.path.dirname(os.path.dirname(__file__))).EquirectangularProcessor
process = psutil.Process()
results = []
for width in map(int, sys.argv[1:] or ['2048']):
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.01): peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=monitor); thread.start()
    image = np.full((width//2, width, 3), .5, np.float32)
    started = time.perf_counter()
    output = P.rotate_equirectangular(image, 23, 12, 5, interpolation='lanczos', use_tiling=True, tile_size=256)
    elapsed = time.perf_counter() - started
    peak[0] = max(peak[0], process.memory_info().rss)
    stop.set(); thread.join()
    assert np.max(abs(output-.5)) < 1e-5
    result = dict(width=width, height=width//2, seconds=elapsed, process_peak_rss_gib=peak[0]/2**30, input_output_gib=image.nbytes*2/2**30)
    results.append(result); print(json.dumps(result), flush=True)
    del image, output
print(json.dumps(dict(platform=platform.platform(), results=results), indent=2))
