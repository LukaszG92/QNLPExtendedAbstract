#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-runs}"
CACHEDIR="${2:-cache}"

python tech01_angle_linear.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR"
python tech02_angle_full_entanglement.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR"
python tech03_angle_reuploading.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR" --L 3
python tech04_rich_readout.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR"
python tech05_amplitude_hashing.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR" --q_amp 6
python tech06_light_compression.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR" --k_base 0 --q_amp 6
python tech07_block_entanglement.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR" --block_size 2
python tech08_morphte_char.py --outdir "$OUTDIR" --cache_dir "$CACHEDIR"
