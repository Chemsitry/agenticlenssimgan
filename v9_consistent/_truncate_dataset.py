"""One-shot: truncate every .npy in full_5000_partial from 5400 → 4050 rows.
In-place header patch + ftruncate (no extra disk required)."""
import os
import re
import gc
import numpy as np
from pathlib import Path

OUTDIR = Path('/Users/nathankvinnesland/Desktop/data prep v9_consistent/output/full_5000_partial')
OLD_N, NEW_N = 5400, 4050

files = ['completed_mask.npy', 'lensed.npy', 'theta_Es.npy', 'z_lens.npy', 'z_source.npy',
         'sigma_v.npy', 'masses.npy',
         'photom_F115W.npy', 'photom_F150W.npy', 'photom_F277W.npy',
         'sigma_v_F115W.npy', 'sigma_v_F150W.npy', 'sigma_v_F277W.npy']
for b in ('F115W', 'F150W', 'F277W', 'F444W'):
    files += [f'images_{b}.npy', f'arcs_{b}.npy', f'galaxies_{b}.npy']

print(f'Target: {OUTDIR}')
print(f'Truncate {OLD_N} → {NEW_N} rows on {len(files)} files\n')

# Pass 1: validate every file (don't touch anything yet)
plans = []
for fn in files:
    p = OUTDIR / fn
    if not p.exists():
        print(f'  MISSING {fn}'); continue
    a = np.load(p, mmap_mode='r')
    shape, dtype, ndim = a.shape, a.dtype, a.ndim
    del a; gc.collect()
    if shape[0] != OLD_N:
        print(f'  SKIP {fn}: shape[0]={shape[0]} (expected {OLD_N})'); continue
    row_bytes = dtype.itemsize * int(np.prod(shape[1:])) if ndim > 1 else dtype.itemsize
    with open(p, 'rb') as f:
        assert f.read(6) == b'\x93NUMPY', f'{fn}: not an npy file'
        major, minor = f.read(1)[0], f.read(1)[0]
        hlen_bytes = 2 if major == 1 else 4
        hlen = int.from_bytes(f.read(hlen_bytes), 'little')
        header_start = f.tell()
        header = f.read(hlen).decode('latin-1')
    new_header = re.sub(rf"'shape':\s*\(\s*{OLD_N}\s*,", f"'shape': ({NEW_N},", header, count=1)
    if new_header == header:
        print(f'  ERROR {fn}: shape regex did not match. Header: {header!r}'); continue
    if len(new_header) != len(header):
        print(f'  ERROR {fn}: header length changed {len(header)} → {len(new_header)}'); continue
    data_start = header_start + hlen
    new_size = data_start + NEW_N * row_bytes
    plans.append({'path': p, 'header_start': header_start, 'new_header': new_header,
                  'new_size': new_size, 'old_size': p.stat().st_size, 'shape': shape, 'dtype': dtype})

if len(plans) != len(files):
    print(f'\nValidation failed for {len(files) - len(plans)} files — aborting.')
    raise SystemExit(1)

print(f'All {len(plans)} files validated. Executing...\n')

# Pass 2: execute
saved_total = 0
for pl in plans:
    p = pl['path']
    with open(p, 'rb+') as f:
        f.seek(pl['header_start'])
        f.write(pl['new_header'].encode('latin-1'))
        f.flush()
        os.ftruncate(f.fileno(), pl['new_size'])
    # Verify
    a = np.load(p, mmap_mode='r')
    expected = (NEW_N,) + pl['shape'][1:]
    assert a.shape == expected, f'{p.name}: post-truncate shape {a.shape} != {expected}'
    del a; gc.collect()
    saved = pl['old_size'] - pl['new_size']
    saved_total += saved
    print(f'  {p.name}: {pl["old_size"]/1e6:7.1f} → {pl["new_size"]/1e6:7.1f} MB  (-{saved/1e6:.1f})')

print(f'\nTotal disk freed: {saved_total/1e9:.2f} GB')
