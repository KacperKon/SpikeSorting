"""Mirror each run's finished KS4 output folder to shared/archival storage.

Run after pipeline_report.py, once sorting/curation/report generation for a
run is done. For each run/probe in the config, copies:

  {output_dir}/catgt_{run}_g{gate}/{run}_g{gate}_imec{prb}/imec{prb}_ks4/

(report PDF, SpikeInterface analyzer, Bombcell output, KS4 sorter_output) and

  {catgt_read|catgt_write|catgt_local}/catgt_{run}_g{gate}/{run}_g{gate}_imec{prb}/
    {run}_g{gate}_tcat.imec{prb}.ap.meta

(metadata for the raw CatGT-filtered AP recording used as sorter input --
just the .meta, not the multi-GB .ap.bin itself) to:

  ephys_sorted_dir : local/mounted path, e.g. a CIFS share -- synced with
                     rsync, mirroring the same catgt_.../imec.._ks4 sub-path
                     so it lines up with other sessions already archived there.

If ephys_sorted_dir isn't set, nothing is synced. One run/probe's failure
doesn't stop the rest -- failures are collected and reported at the end.

Usage:
  python sync_outputs.py config.yaml
"""

import sys
import subprocess
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ks4_dir(run, prb, config):
    # Mirrors ks4_dir() in pipeline_ks4.py -- kept separate so this script
    # doesn't need spikeinterface installed to run.
    run_str = f"{run['name']}_g{run['gate']}"
    return Path(config['output_dir']) / f"catgt_{run_str}" / f"{run_str}_imec{prb}" / f"imec{prb}_ks4"


def relative_prb_path(run, prb):
    run_str = f"{run['name']}_g{run['gate']}"
    return f"catgt_{run_str}/{run_str}_imec{prb}"


def relative_ks4_path(run, prb):
    return f"{relative_prb_path(run, prb)}/imec{prb}_ks4"


def catgt_prb_dir(run, prb, config):
    # Mirrors _catgt_root()/catgt_prb_dir() in pipeline_ks4.py -- finds
    # wherever the raw CatGT-filtered output actually lives: catgt_local
    # (if copy_raw_to_local pulled a copy there), else catgt_read, else
    # catgt_write.
    prb_subdir = relative_prb_path(run, prb)
    if config.get('copy_raw_to_local') and config.get('catgt_local'):
        if (Path(config['catgt_local']) / prb_subdir).exists():
            return Path(config['catgt_local']) / prb_subdir
    if config.get('catgt_read') and (Path(config['catgt_read']) / prb_subdir).exists():
        return Path(config['catgt_read']) / prb_subdir
    return Path(config['catgt_write']) / prb_subdir


def catgt_ap_meta_path(run, prb, config):
    # Just the .meta sidecar -- the .ap.bin itself is multi-GB and not archived.
    run_str = f"{run['name']}_g{run['gate']}"
    return catgt_prb_dir(run, prb, config) / f"{run_str}_tcat.imec{prb}.ap.meta"


def sync_to_ephys_sorted(local_dir, ephys_sorted_root, rel_path):
    dest = Path(ephys_sorted_root) / rel_path
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [Sync] rsync -> {dest}")
    subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{local_dir}/", f"{dest}/"],
        check=True,
    )


def sync_file_to_ephys_sorted(local_file, ephys_sorted_root, rel_path):
    dest = Path(ephys_sorted_root) / rel_path
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [Sync] rsync {local_file.name} -> {dest}")
    subprocess.run(
        ["rsync", "-a", "--info=progress2", str(local_file), f"{dest}/"],
        check=True,
    )


def sync_run_probe(run, prb, config):
    local_dir = ks4_dir(run, prb, config)
    ephys_sorted_root = config.get('ephys_sorted_dir')

    if not ephys_sorted_root:
        print("  No ephys_sorted_dir configured, nothing to do.")
        return

    if local_dir.exists():
        sync_to_ephys_sorted(local_dir, ephys_sorted_root, relative_ks4_path(run, prb))
    else:
        print(f"  [{run['name']} probe {prb}] No KS4 output at {local_dir}, skipping.")

    ap_meta = catgt_ap_meta_path(run, prb, config)
    if ap_meta.exists():
        sync_file_to_ephys_sorted(ap_meta, ephys_sorted_root, relative_prb_path(run, prb))
    else:
        print(f"  [{run['name']} probe {prb}] No raw .ap.meta found at {ap_meta}, skipping.")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)

    failed = []
    for run in config['runs']:
        print(f"\n[{run['name']}]")
        for prb in run['probes']:
            try:
                sync_run_probe(run, prb, config)
            except Exception as exc:
                print(f"  [ERROR] probe {prb}: {exc}", file=sys.stderr)
                failed.append(f"{run['name']} probe {prb}")

    if failed:
        print(f"\nSync failed for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
