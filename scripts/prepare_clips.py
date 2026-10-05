"""Pre-compute training clips. Thin wrapper around :mod:`wilrosync.data.prepare` (see its docstring).

Example:
    python scripts/prepare_clips.py --manifest data/wild.jsonl --root data/raw --out data/latents/clips \
        --size 512 512 --frames 49 --clips-per-video 4
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wilrosync.data.prepare import Encoders, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
