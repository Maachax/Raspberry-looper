"""Feasibility benchmark for pedalboard on this machine.

Renders a representative loop through a 3-effect chain and measures per-render
time, then measures per-block time for a live reverb. Run on the Pi:
    python tools/fx_bench.py
"""
import time
import numpy as np
from pedalboard import Pedalboard, Reverb, Delay, Distortion

SR = 44100
BLOCK = 256


def main():
    dry = np.random.randn(SR * 8).astype(np.float32) * 0.2   # 8s mono
    chain = Pedalboard([Distortion(drive_db=12), Delay(delay_seconds=0.25, feedback=0.4, mix=0.4), Reverb(room_size=0.5)])

    # Offline render (tiled x3, like render_wet)
    tiled = np.tile(dry, 3)
    t0 = time.perf_counter()
    for _ in range(5):
        chain(tiled, SR, reset=True)
    render_ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"render_wet (8s x3, 3 fx): {render_ms:.1f} ms  (target: < 100 ms)")

    # Live bus reverb per block
    bus = Pedalboard([Reverb(room_size=0.6)])
    block = np.random.randn(BLOCK).astype(np.float32) * 0.2
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        bus(block, SR, reset=False)
    per_block_ms = (time.perf_counter() - t0) / n * 1000
    budget_ms = BLOCK / SR * 1000
    print(f"live bus reverb / block: {per_block_ms:.3f} ms  (block budget: {budget_ms:.2f} ms)")
    print("PASS" if render_ms < 100 and per_block_ms < budget_ms * 0.5 else "REVIEW")


if __name__ == "__main__":
    main()
