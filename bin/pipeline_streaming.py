"""Run the full pipeline one recording at a time, to keep local disk use low.

Instead of running every stage over all runs (sorting for all runs, then
curation for all runs, ...), each run goes through every stage before moving on:

  download -> sorting (pipeline_ks4) -> curation -> report -> sync_outputs
           -> upload remaining local files to scratch -> verify -> delete local copy

While run N is being processed, run N+1 is prepared in a background thread:
download_raw_data.py fetches it to npx_directory if needed (legacy servers),
then prefetch_raw copies it to raw_local. So at most two runs are on
local disk at any time.

A run's local files (raw_local copy, CatGT output in catgt_write/catgt_local,
KS4/analyzer/report output in output_dir) are deleted ONLY when:
  * every stage exited with code 0, and
  * everything was copied to ephys_sorted_dir and every file there was checked
    to have the same size as the local file.
Otherwise the local files are kept and the run is listed as failed at the end.

What goes to ephys_sorted_dir/catgt_{run}_g{gate}/:
  * the whole output_dir/catgt_{run}_g{gate}/ folder (imec*_ks4 etc.)
  * the CatGT output folder, EXCEPT *.bin files (the multi-GB filtered
    recording), unless sync_catgt_bin: true. This keeps the small CatGT files
    (sync / NI event .txt, .meta) that would otherwise be lost on deletion.

Config keys (on top of the normal ones):
  ephys_sorted_dir  : required -- where the run is uploaded before deletion.
  stream_cleanup    : default true. Set false to run per-run but keep local files.
  sync_catgt_bin    : default false. Also upload the CatGT .bin files to scratch.

Run this inside the si_ks4 env. Curation is run in the `curation` env
(override with curation_env / sorting_env in the config).

Usage:
  python pipeline_streaming.py config.yaml
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml

import download_raw_data
import pipeline_ks4 as pk

SCRIPT_DIR = Path(__file__).resolve().parent


def _ts():
    return pk._ts()


def _run_str(run):
    return f"{run['name']}_g{run['gate']}"


def _micromamba():
    exe = os.environ.get('MAMBA_EXE') or shutil.which('micromamba')
    if not exe:
        raise RuntimeError("micromamba not found (MAMBA_EXE unset and not on PATH).")
    return exe


def run_stage(script, env, cfg_path):
    """Run one pipeline script on a single-run config. Returns its exit code."""
    cmd = [_micromamba(), 'run', '-n', env, 'python', str(SCRIPT_DIR / script), str(cfg_path)]
    print(f"\n[{_ts()}] [Streaming] >>> {script} ({env})", flush=True)
    rc = subprocess.call(cmd, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    print(f"[{_ts()}] [Streaming] <<< {script} exit code {rc}", flush=True)
    return rc


def write_run_config(config, run, cfg_dir):
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{_run_str(run)}.yaml"
    with open(path, 'w') as f:
        yaml.safe_dump({**config, 'runs': [run]}, f, sort_keys=False)
    return path


# --- Background preparation of the next run ---

def prepare_run(run, config, reserve_bytes):
    """Make sure raw data is on npx_directory, then copy it to raw_local. Never raises."""
    run_str = _run_str(run)
    try:
        download_raw_data.ensure_run_downloaded(run, config)
    except BaseException as e:
        print(f"  [{_ts()}] [Download] {run_str} failed: {e!r}. Sorting will likely fail for this run.")
        return
    if config.get('prefetch_raw') and config.get('raw_local'):
        pk.prefetch_raw(run, config, reserve_bytes)


def _reserve_for(run, config):
    # Same logic as pipeline_ks4.main: room the current run's CatGT output still needs.
    if not config.get('raw_local'):
        return 0
    cw, rl = Path(config['catgt_write']), Path(config['raw_local'])
    same_disk = cw.exists() and rl.exists() and cw.stat().st_dev == rl.stat().st_dev
    if same_disk and (config.get('force_rerun_catgt') or not all(
            pk.catgt_bin_path(run, prb, config).exists() for prb in run['probes'])):
        return pk._raw_ap_bytes(run, config)
    return 0


# --- Upload + verify + delete ---

def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _protected_roots(config):
    # Never delete anything living under shared/source storage.
    return [config[k] for k in ('ephys_sorted_dir', 'catgt_read', 'npx_directory') if config.get(k)]


def local_run_dirs(run, config):
    """Local folders holding this run's data that can be uploaded + deleted.
    Returns list of (local_dir, exclude_bins)."""
    run_str = _run_str(run)
    exclude_bins = not config.get('sync_catgt_bin', False)
    candidates = [(Path(config['output_dir']) / f"catgt_{run_str}", False)]
    for key in ('catgt_write', 'catgt_local'):
        if config.get(key):
            candidates.append((Path(config[key]) / f"catgt_{run_str}", exclude_bins))

    seen, out = set(), []
    protected = _protected_roots(config)
    for d, excl in candidates:
        if not d.exists():
            continue
        r = d.resolve()
        if r in seen:
            continue
        if any(_is_within(r, p) for p in protected):
            print(f"  [Cleanup] {d} is on shared storage, leaving it alone.")
            continue
        seen.add(r)
        out.append((d, excl))
    return out


def upload_dir(local_dir, dest_dir, exclude_bins):
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ['rsync', '-rt', '--info=progress2']
    if exclude_bins:
        cmd += ['--exclude=*.bin']
    cmd += [f"{local_dir}/", f"{dest_dir}/"]
    print(f"  [Upload] {local_dir} -> {dest_dir}" + (" (without *.bin)" if exclude_bins else ""))
    subprocess.run(cmd, check=True)


def verify_dir(local_dir, dest_dir, exclude_bins):
    """Every local file (minus excluded .bin) must exist at dest with the same size."""
    bad = []
    n = 0
    for f in local_dir.rglob('*'):
        if not f.is_file() or (exclude_bins and f.suffix == '.bin'):
            continue
        n += 1
        g = dest_dir / f.relative_to(local_dir)
        if not g.is_file() or g.stat().st_size != f.stat().st_size:
            bad.append(str(f.relative_to(local_dir)))
    if bad:
        raise RuntimeError(f"{len(bad)}/{n} file(s) missing or different on scratch, e.g. {bad[:5]}")
    print(f"  [Verify] {n} file(s) from {local_dir} confirmed on scratch.")


def upload_and_cleanup(run, config):
    run_str = _run_str(run)
    dest = Path(config['ephys_sorted_dir']) / f"catgt_{run_str}"
    dirs = local_run_dirs(run, config)

    for d, excl in dirs:
        upload_dir(d, dest, excl)
    for d, excl in dirs:
        verify_dir(d, dest, excl)

    if not config.get('stream_cleanup', True):
        print(f"  [Cleanup] stream_cleanup is false, keeping local files for {run_str}.")
        return
    for d, _ in dirs:
        shutil.rmtree(d)
        print(f"  [Cleanup] Deleted local {d}")
    if config.get('raw_local'):
        pk.clear_raw_local(run, config)


# --- Main loop ---

def process_one(run, config, cfg_dir):
    """Run all stages for one run. Returns None on success, else a failure reason."""
    cfg_path = write_run_config(config, run, cfg_dir)
    sort_env = config.get('sorting_env', 'si_ks4')
    cur_env = config.get('curation_env', 'curation')

    if run_stage('pipeline_ks4.py', sort_env, cfg_path) != 0:
        return 'sorting failed'
    failures = []
    if run_stage('pipeline_curation.py', cur_env, cfg_path) != 0:
        failures.append('curation')
    if run_stage('pipeline_report.py', sort_env, cfg_path) != 0:
        failures.append('report')
    if run_stage('sync_outputs.py', sort_env, cfg_path) != 0:
        failures.append('sync_outputs')
    if failures:
        return f"{', '.join(failures)} failed, local files kept"

    try:
        upload_and_cleanup(run, config)
    except Exception as e:
        return f"upload/verify failed ({e!r}), local files kept"
    return None


def main():
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else 'config.yaml').resolve()
    config = pk.load_config(config_path)
    if not config.get('ephys_sorted_dir'):
        raise ValueError("pipeline_streaming.py needs ephys_sorted_dir set (upload target before deleting).")
    if config.get('prefetch_raw') and not config.get('raw_local'):
        raise ValueError("prefetch_raw is true but raw_local is not set in config.")

    cfg_dir = Path('logs') / 'streaming_configs' / config_path.stem
    runs = config['runs']
    failed = []

    # First run: make sure its raw data is reachable; it is read from the network
    # (as in pipeline_ks4.py), so no local prefetch for it.
    try:
        download_raw_data.ensure_run_downloaded(runs[0], config)
    except BaseException as e:
        print(f"[Download] {_run_str(runs[0])} failed: {e!r}")

    pending = None
    for i, run in enumerate(runs):
        if pending is not None:
            if pending.is_alive():
                print(f"\n[{_ts()}] [Streaming] Waiting for download/prefetch of {_run_str(run)}...")
            pending.join()
        pending = None
        if i + 1 < len(runs):
            pending = threading.Thread(
                target=prepare_run, args=(runs[i + 1], config, _reserve_for(run, config)), daemon=True)
            pending.start()

        print(f"\n{'#'*60}\n[{_ts()}] [Streaming] Run {i+1}/{len(runs)}: {_run_str(run)}\n{'#'*60}", flush=True)
        reason = process_one(run, config, cfg_dir)
        if reason:
            print(f"[{_ts()}] [Streaming] {_run_str(run)}: {reason}")
            failed.append((_run_str(run), reason))
        else:
            print(f"[{_ts()}] [Streaming] {_run_str(run)} done, uploaded and cleaned up.")

    if failed:
        print(f"\n[{_ts()}] Runs that did not complete (local data kept):")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        sys.exit(1)
    print(f"\n[{_ts()}] All {len(runs)} runs complete.")


if __name__ == '__main__':
    main()
