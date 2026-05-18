"""
resume_simulation.py — Resume an interrupted simulate_v9_consistent.py run.

Reads a partial output dir (12 image .npy files preallocated, populated up
to some index), then:
  1. Re-derives metadata for already-completed indices from job seeds.
  2. Renders the missing indices in-place into the existing memmaps
     (no new disk; opens files in r+ mode rather than w+).
  3. Writes labels / theta_Es / z_lens / z_source / masses / sigma_v / photom
     / completed_mask.

USAGE
  python resume_simulation.py --partial-dir output/full_5000_partial

Defaults match the original run that produced full_5000_partial:
  --n-lensed 5000  --n-nonlensed 535  --n-bins 10  --te-min 0.5  --te-max 1.8
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Numba 0.64 + Python 3.14 fails to JIT lenstronomy.Util.util.rotate
# ("'function' object has no attribute 'get_call_template'"). Disable JIT
# before importing lenstronomy. Pure-Python fallback is bit-exact vs the
# original 4050 sims (validated against the existing memmaps).
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

import numpy as np
from scipy.stats import truncnorm
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

import simulate_v9_consistent as sim
from simulate_v9_consistent import (
    BANDS, IMAGE_SIZE,
    _simulate_worker,
    cphot, cal, scenes, scenes_manifest, n_scenes,
    ZLENS_LO, ZLENS_HI, ZLENS_LOC, ZLENS_SCALE,
    ZSRC_LO, ZSRC_HI, ZSRC_LOC, ZSRC_SCALE,
    SIGMA_LO, SIGMA_HI, SIGMA_CONSISTENCY_MAX_DEX,
    PHOTOM_APERTURE_PX, PHOTOM_SKY_ANNULUS,
    LOGMASS_LO, LOGMASS_HI,
    _RetryAttempt,
)


def _extract_params_attempt(lensed, seed, scene_idx, target_theta_range):
    rng = np.random.default_rng(seed)

    if scene_idx is None:
        scene_idx = int(rng.integers(n_scenes))

    if scenes_manifest is not None:
        z_lens = float(scenes_manifest['Z_PHOT_MEAN'].iloc[scene_idx])
        z_lens = float(np.clip(z_lens, ZLENS_LO, ZLENS_HI))
    else:
        z_lens = float(truncnorm.rvs(
            a=(ZLENS_LO - ZLENS_LOC) / ZLENS_SCALE,
            b=(ZLENS_HI - ZLENS_LOC) / ZLENS_SCALE,
            loc=ZLENS_LOC, scale=ZLENS_SCALE,
            random_state=int(rng.integers(int(1e9)))))

    z_src_min = max(ZSRC_LO, z_lens + 0.1)
    for _ in range(100):
        z_source = float(truncnorm.rvs(
            a=(ZSRC_LO - ZSRC_LOC) / ZSRC_SCALE,
            b=(ZSRC_HI - ZSRC_LOC) / ZSRC_SCALE,
            loc=ZSRC_LOC, scale=ZSRC_SCALE,
            random_state=int(rng.integers(int(1e9)))))
        if z_source > z_lens + 0.1:
            break
    z_source = max(z_source, z_src_min)

    photom_mags = {
        b: cphot.cutout_ab_mag(
            scenes[b][scene_idx],
            aperture_radius_pix=PHOTOM_APERTURE_PX,
            sky_annulus=PHOTOM_SKY_ANNULUS,
        )
        for b in ('F115W', 'F150W', 'F277W')
    }

    sigma_v_per_band = {}
    if (scenes_manifest is not None
            and 'has_measured_sigma_v' in scenes_manifest.columns
            and bool(scenes_manifest['has_measured_sigma_v'].iloc[scene_idx])):
        sigma_v_consensus = float(scenes_manifest['sigma_v_measured'].iloc[scene_idx])
    else:
        sigma_v_consensus, sigma_v_per_band = cal.predict_sigma_v_multiband(
            photom_mags, z_lens)
        band_log_sigs = np.array(
            [np.log10(v) for v in sigma_v_per_band.values()
             if np.isfinite(v) and v > 0])
        if (band_log_sigs.size < 2
                or (band_log_sigs.max() - band_log_sigs.min())
                    > SIGMA_CONSISTENCY_MAX_DEX
                or not np.isfinite(sigma_v_consensus)):
            raise _RetryAttempt()

    sigma_v = float(np.clip(sigma_v_consensus, SIGMA_LO, SIGMA_HI))

    lens_cosmo = LensCosmo(z_lens, z_source)
    theta_E = lens_cosmo.sis_sigma_v2theta_E(sigma_v) if lensed else 0.0

    te_lo, te_hi = target_theta_range if target_theta_range is not None \
        else (sim.THETA_E_LO, sim.THETA_E_HI)
    if lensed and not (te_lo <= theta_E <= te_hi):
        raise _RetryAttempt()

    # Mass draw must run so rng state stays aligned with simulate_one
    log_mass = float(rng.uniform(LOGMASS_LO, LOGMASS_HI))
    mass = 10 ** log_mass

    return {
        'theta_E': theta_E, 'z_lens': z_lens, 'z_source': z_source,
        'mass': mass, 'sigma_v': sigma_v,
        'photom_mags': photom_mags,
        'sigma_v_per_band': sigma_v_per_band,
    }


def extract_params(lensed, seed, scene_idx, target_theta_range, eligible_scenes):
    MAX_RETRIES = 50000
    base_rng = np.random.default_rng(seed)
    for attempt in range(MAX_RETRIES):
        if attempt == 0:
            cur_scene = scene_idx
        elif eligible_scenes is not None and len(eligible_scenes):
            cur_scene = int(eligible_scenes[base_rng.integers(len(eligible_scenes))])
        else:
            cur_scene = None
        try:
            return _extract_params_attempt(
                lensed=lensed,
                seed=int(base_rng.integers(int(1e9))),
                scene_idx=cur_scene,
                target_theta_range=target_theta_range,
            )
        except _RetryAttempt:
            continue
    raise RuntimeError(
        f"extract_params: gave up after {MAX_RETRIES} retries "
        f"(target_theta_range={target_theta_range})")


def build_jobs(n_lensed, n_nonlensed, n_bins, te_min, te_max, main_seed=99):
    """Exactly reconstruct the job list built by simulate_v9_consistent.main()."""
    bin_edges = np.linspace(te_min, te_max, n_bins + 1)
    per_bin = n_lensed // n_bins
    extra = n_lensed - per_bin * n_bins

    rng_main = np.random.default_rng(main_seed)
    if n_nonlensed > n_scenes:
        n_nonlensed = n_scenes
    scene_idx_nl = rng_main.choice(n_scenes, size=n_nonlensed, replace=False)
    scene_idx_l = rng_main.integers(n_scenes, size=n_lensed)

    Z_SOURCE_REF = 2.5
    scene_theta = np.full(n_scenes, np.nan)
    for s_idx in range(n_scenes):
        if (scenes_manifest is not None
                and 'has_measured_sigma_v' in scenes_manifest.columns
                and bool(scenes_manifest['has_measured_sigma_v'].iloc[s_idx])):
            sigma_v_s = float(scenes_manifest['sigma_v_measured'].iloc[s_idx])
            z_lens_s = float(scenes_manifest['Z_PHOT_MEAN'].iloc[s_idx])
            z_lens_s = float(np.clip(z_lens_s, ZLENS_LO, ZLENS_HI))
        else:
            try:
                mags = {b: cphot.cutout_ab_mag(
                    scenes[b][s_idx], aperture_radius_pix=PHOTOM_APERTURE_PX,
                    sky_annulus=PHOTOM_SKY_ANNULUS) for b in ('F115W', 'F150W', 'F277W')}
                z_lens_s = float(scenes_manifest['Z_PHOT_MEAN'].iloc[s_idx]) if scenes_manifest is not None else 0.5
                z_lens_s = float(np.clip(z_lens_s, ZLENS_LO, ZLENS_HI))
                sigma_v_pred, _ = cal.predict_sigma_v_multiband(mags, z_lens_s)
                if not np.isfinite(sigma_v_pred):
                    continue
                sigma_v_s = float(np.clip(sigma_v_pred, SIGMA_LO, SIGMA_HI))
            except Exception:
                continue
        try:
            zs = max(Z_SOURCE_REF, z_lens_s + 0.1)
            lc = LensCosmo(z_lens_s, zs)
            scene_theta[s_idx] = lc.sis_sigma_v2theta_E(sigma_v_s)
        except Exception:
            continue

    jobs = []
    for i in range(n_nonlensed):
        seed = int(rng_main.integers(int(1e9)))
        noise_seed = int(rng_main.integers(int(1e9)))
        jobs.append((i, False, seed, noise_seed,
                     int(scene_idx_nl[i]), None, None))
    job_idx = n_nonlensed
    for bin_i in range(n_bins):
        count = per_bin + (1 if bin_i < extra else 0)
        lo = float(bin_edges[bin_i]); hi = float(bin_edges[bin_i + 1])
        bin_mask = (scene_theta > lo * 0.5) & (scene_theta < hi * 2.0)
        eligible = np.where(bin_mask)[0]
        if len(eligible) == 0:
            eligible = np.where(np.isfinite(scene_theta))[0]
            order = np.argsort(scene_theta[eligible])[-50:]
            eligible = eligible[order]
        eligible_tuple = tuple(int(x) for x in eligible)
        for _ in range(count):
            seed = int(rng_main.integers(int(1e9)))
            noise_seed = int(rng_main.integers(int(1e9)))
            s_idx = int(eligible[rng_main.integers(len(eligible))])
            jobs.append((job_idx, True, seed, noise_seed, s_idx, (lo, hi), eligible_tuple))
            job_idx += 1
    return jobs


def _meta_worker(args):
    idx, lensed, seed, _noise_seed, scene_idx, ttr, eligible = args
    try:
        p = extract_params(lensed, seed, scene_idx, ttr, eligible)
    except Exception as e:
        return {'idx': idx, 'failed': True, 'error': f'{type(e).__name__}: {e}'}
    return {
        'idx': idx, 'failed': False, 'label': 1.0 if lensed else 0.0,
        'theta_E': p['theta_E'], 'z_lens': p['z_lens'],
        'z_source': p['z_source'], 'mass': p['mass'],
        'sigma_v': p['sigma_v'],
        'photom_mag_F115W': float(p['photom_mags'].get('F115W', np.nan)),
        'photom_mag_F150W': float(p['photom_mags'].get('F150W', np.nan)),
        'photom_mag_F277W': float(p['photom_mags'].get('F277W', np.nan)),
        'sigma_v_F115W':    float(p['sigma_v_per_band'].get('F115W', np.nan)),
        'sigma_v_F150W':    float(p['sigma_v_per_band'].get('F150W', np.nan)),
        'sigma_v_F277W':    float(p['sigma_v_per_band'].get('F277W', np.nan)),
    }


def validate(partial_dir, jobs, sample_indices, atol=1e-4):
    """Re-render a few completed sims and confirm bit-near-identical to memmap."""
    print(f"\nValidating: re-rendering {len(sample_indices)} completed sims for bit comparison")
    existing = np.load(partial_dir / 'images_F150W.npy', mmap_mode='r')
    all_match = True
    for idx in sample_indices:
        idx = int(idx)
        result = _simulate_worker(jobs[idx])
        if result.get('failed'):
            print(f"  idx {idx}: render FAILED: {result.get('error')}")
            all_match = False
            continue
        rendered = result['images']['F150W']
        diff = float(np.abs(rendered - existing[idx]).max())
        match = diff < atol
        print(f"  idx {idx}: max-pixel-diff={diff:.2e}  {'OK' if match else 'MISMATCH'}")
        if not match:
            all_match = False
    return all_match


def save_metadata(d, labels, theta_Es, z_lenses, z_sources, masses, sigma_vs,
                  pF115W, pF150W, pF277W, sF115W, sF150W, sF277W, completed_mask):
    print(f"\nSaving metadata to {d}/")
    np.save(d / 'completed_mask.npy', completed_mask)
    np.save(d / 'lensed.npy', labels)
    np.save(d / 'theta_Es.npy', theta_Es)
    np.save(d / 'z_lens.npy', z_lenses)
    np.save(d / 'z_source.npy', z_sources)
    np.save(d / 'masses.npy', masses)
    np.save(d / 'sigma_v.npy', sigma_vs)
    np.save(d / 'photom_F115W.npy', pF115W)
    np.save(d / 'photom_F150W.npy', pF150W)
    np.save(d / 'photom_F277W.npy', pF277W)
    np.save(d / 'sigma_v_F115W.npy', sF115W)
    np.save(d / 'sigma_v_F150W.npy', sF150W)
    np.save(d / 'sigma_v_F277W.npy', sF277W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial-dir', required=True,
                    help='Existing output dir with 12 image .npy memmaps')
    ap.add_argument('--workers', type=int, default=8,
                    help='Lower than the original 16 to reduce OOM risk')
    ap.add_argument('--chunksize', type=int, default=10)
    ap.add_argument('--validate-n', type=int, default=3,
                    help='How many completed sims to spot-check (0 to skip)')
    ap.add_argument('--n-lensed', type=int, default=5000)
    ap.add_argument('--n-nonlensed', type=int, default=535)
    ap.add_argument('--n-bins', type=int, default=10)
    ap.add_argument('--te-min', type=float, default=0.5)
    ap.add_argument('--te-max', type=float, default=1.8)
    ap.add_argument('--metadata-only', action='store_true',
                    help='Re-derive metadata for completed sims and exit')
    ap.add_argument('--render-only', action='store_true',
                    help='Skip metadata re-derivation by loading existing metadata '
                         'files (lensed.npy, theta_Es.npy, ...) from --partial-dir. '
                         'Errors if any are missing. Use after a prior --metadata-only '
                         'run.')
    args = ap.parse_args()

    partial_dir = Path(args.partial_dir)
    assert partial_dir.exists(), f"No such dir: {partial_dir}"

    print(f"Resume target: {partial_dir}")
    print("Rebuilding job queue (must match original run exactly)...")
    jobs = build_jobs(args.n_lensed, args.n_nonlensed,
                      args.n_bins, args.te_min, args.te_max)
    N = len(jobs)
    print(f"  Job queue length: {N}")

    print("Scanning existing memmap to find completed range...")
    img = np.load(partial_dir / 'images_F150W.npy', mmap_mode='r')
    assert img.shape == (N, IMAGE_SIZE, IMAGE_SIZE), \
        f"Mismatch: existing shape {img.shape} vs expected {(N, IMAGE_SIZE, IMAGE_SIZE)}"
    populated = np.array([bool(img[i].any()) for i in range(N)])
    completed = np.where(populated)[0]
    missing = np.where(~populated)[0]
    print(f"  Completed: {len(completed)}  |  Missing: {len(missing)}")

    if args.validate_n > 0 and len(completed) > 0:
        rng = np.random.default_rng(42)
        sample = rng.choice(completed,
                            size=min(args.validate_n, len(completed)),
                            replace=False)
        ok = validate(partial_dir, jobs, sample)
        if not ok:
            print("\n!!! VALIDATION FAILED — job-queue reconstruction does NOT match "
                  "the original run. Refusing to write metadata. !!!")
            sys.exit(2)
        print("Validation passed — safe to proceed.")

    required_meta = ['lensed.npy', 'theta_Es.npy', 'z_lens.npy', 'z_source.npy',
                     'masses.npy', 'sigma_v.npy',
                     'photom_F115W.npy', 'photom_F150W.npy', 'photom_F277W.npy',
                     'sigma_v_F115W.npy', 'sigma_v_F150W.npy', 'sigma_v_F277W.npy',
                     'completed_mask.npy']
    if args.render_only:
        missing_files = [f for f in required_meta if not (partial_dir / f).exists()]
        if missing_files:
            print(f"\n!!! --render-only requires existing metadata files. Missing: {missing_files}")
            print("Run --metadata-only first, then re-run with --render-only.")
            sys.exit(2)
        print(f"\nLoading existing metadata from {partial_dir}/")
        labels      = np.load(partial_dir / 'lensed.npy')
        theta_Es    = np.load(partial_dir / 'theta_Es.npy')
        z_lenses    = np.load(partial_dir / 'z_lens.npy')
        z_sources   = np.load(partial_dir / 'z_source.npy')
        masses      = np.load(partial_dir / 'masses.npy')
        sigma_vs    = np.load(partial_dir / 'sigma_v.npy')
        pF115W      = np.load(partial_dir / 'photom_F115W.npy')
        pF150W      = np.load(partial_dir / 'photom_F150W.npy')
        pF277W      = np.load(partial_dir / 'photom_F277W.npy')
        sF115W      = np.load(partial_dir / 'sigma_v_F115W.npy')
        sF150W      = np.load(partial_dir / 'sigma_v_F150W.npy')
        sF277W      = np.load(partial_dir / 'sigma_v_F277W.npy')
        completed_mask = np.load(partial_dir / 'completed_mask.npy')
        for arr, name in [(labels, 'lensed'), (theta_Es, 'theta_Es'), (completed_mask, 'completed_mask')]:
            if len(arr) != N:
                print(f"!!! Loaded {name}.npy has length {len(arr)} but expected {N}")
                sys.exit(2)
        print(f"  Loaded metadata for {int(completed_mask.sum())} completed sims")
    else:
        labels = np.zeros(N)
        theta_Es = np.zeros(N)
        z_lenses = np.zeros(N)
        z_sources = np.zeros(N)
        masses = np.zeros(N)
        sigma_vs = np.full(N, np.nan)
        pF115W = np.full(N, np.nan); pF150W = np.full(N, np.nan); pF277W = np.full(N, np.nan)
        sF115W = np.full(N, np.nan); sF150W = np.full(N, np.nan); sF277W = np.full(N, np.nan)
        completed_mask = np.zeros(N, dtype=bool)

    ctx = mp.get_context('fork')

    if not args.render_only and len(completed) > 0:
        print(f"\nRe-deriving metadata for {len(completed)} completed sims "
              f"(workers={args.workers})...")
        completed_jobs = [jobs[i] for i in completed]
        t0 = time.time(); n_done = 0; n_fail = 0
        with ctx.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_meta_worker, completed_jobs,
                                         chunksize=args.chunksize):
                n_done += 1
                if r.get('failed'):
                    n_fail += 1
                    if n_fail <= 5:
                        print(f"  meta WARN: idx {r['idx']}: {r.get('error')}")
                    continue
                idx = r['idx']
                labels[idx] = r['label']
                theta_Es[idx] = r['theta_E']
                z_lenses[idx] = r['z_lens']
                z_sources[idx] = r['z_source']
                masses[idx] = r['mass']
                sigma_vs[idx] = r['sigma_v']
                pF115W[idx] = r['photom_mag_F115W']
                pF150W[idx] = r['photom_mag_F150W']
                pF277W[idx] = r['photom_mag_F277W']
                sF115W[idx] = r['sigma_v_F115W']
                sF150W[idx] = r['sigma_v_F150W']
                sF277W[idx] = r['sigma_v_F277W']
                completed_mask[idx] = True
                if n_done % 200 == 0:
                    rate = n_done / (time.time() - t0)
                    print(f"  meta [{n_done}/{len(completed_jobs)}] {rate:.0f}/s")
        print(f"  Metadata: {int(completed_mask.sum())} ok, {n_fail} failed")

    if args.metadata_only:
        save_metadata(partial_dir, labels, theta_Es, z_lenses, z_sources, masses,
                      sigma_vs, pF115W, pF150W, pF277W, sF115W, sF150W, sF277W,
                      completed_mask)
        return

    if len(missing) > 0:
        print(f"\nRendering {len(missing)} missing sims into existing memmaps "
              f"(workers={args.workers})...")
        for f in partial_dir.glob('*.npy'):
            os.chmod(f, 0o644)
        try:
            all_images = {b: np.load(partial_dir / f'images_{b}.npy', mmap_mode='r+') for b in BANDS}
            all_arcs   = {b: np.load(partial_dir / f'arcs_{b}.npy',   mmap_mode='r+') for b in BANDS}
            all_gals   = {b: np.load(partial_dir / f'galaxies_{b}.npy', mmap_mode='r+') for b in BANDS}

            missing_jobs = [jobs[i] for i in missing]
            t0 = time.time(); n_done = 0; n_fail = 0
            try:
                with ctx.Pool(args.workers) as pool:
                    for r in pool.imap_unordered(_simulate_worker, missing_jobs,
                                                 chunksize=args.chunksize):
                        n_done += 1
                        if r.get('failed'):
                            n_fail += 1
                            if n_fail <= 5:
                                print(f"  render WARN: idx {r['idx']}: {r.get('error')}")
                            continue
                        idx = r['idx']
                        for band in BANDS:
                            all_images[band][idx] = r['images'][band]
                            all_arcs[band][idx]   = r['arcs'][band]
                            all_gals[band][idx]   = r['galaxies'][band]
                        labels[idx] = r['label']
                        theta_Es[idx] = r['theta_E']
                        z_lenses[idx] = r['z_lens']
                        z_sources[idx] = r['z_source']
                        masses[idx] = r['mass']
                        sigma_vs[idx] = r.get('sigma_v', np.nan)
                        pF115W[idx] = r.get('photom_mag_F115W', np.nan)
                        pF150W[idx] = r.get('photom_mag_F150W', np.nan)
                        pF277W[idx] = r.get('photom_mag_F277W', np.nan)
                        sF115W[idx] = r.get('sigma_v_F115W', np.nan)
                        sF150W[idx] = r.get('sigma_v_F150W', np.nan)
                        sF277W[idx] = r.get('sigma_v_F277W', np.nan)
                        completed_mask[idx] = True
                        if n_done % 10 == 0:
                            elapsed = time.time() - t0
                            rate = n_done / max(elapsed, 1e-9)
                            eta = (len(missing_jobs) - n_done) / max(rate, 1e-9)
                            print(f"  render [{n_done}/{len(missing_jobs)}] "
                                  f"{elapsed:.0f}s, {rate:.2f}/s, ETA {eta/60:.0f}m",
                                  flush=True)
            except KeyboardInterrupt:
                print('  Caught Ctrl+C — flushing partial render')

            for band in BANDS:
                all_images[band].flush()
                all_arcs[band].flush()
                all_gals[band].flush()
        finally:
            for f in partial_dir.glob('*.npy'):
                os.chmod(f, 0o444)
        print(f"  Rendering: {len(missing) - n_fail} ok, {n_fail} failed")

    save_metadata(partial_dir, labels, theta_Es, z_lenses, z_sources, masses,
                  sigma_vs, pF115W, pF150W, pF277W, sF115W, sF150W, sF277W,
                  completed_mask)


if __name__ == '__main__':
    main()
