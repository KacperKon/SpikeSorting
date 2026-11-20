"""Make sure raw SpikeGLX data for every run in a config exists under npx_directory.

Run before sorting starts. For each run in config['runs'], checks in order:

  1. npx_directory              -- already there, nothing to do.
  2. Each dir in legacy_raw_dirs -- older lab servers with the same
                                    {name}_g{gate} layout; copied via rsync.

Config fields (see config_example.yaml):
  legacy_raw_dirs    : list of older raw-data roots to check, in order.
  force_rerun_download: re-fetch from legacy_raw_dirs even if the run already
                        exists under npx_directory. If no legacy dir has the
                        run, the existing local copy is kept.

Usage:
  python download_raw_data.py config.yaml
"""

import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def find_run_dir(run, root):
    """Look for this run's folder under `root`, in either SpikeGLX layout.

    New: {root}/{name}_g{gate}/
    Old: {root}/{name}/{name}_g{gate}/
    Returns the run folder Path, or None if not found under this root.
    """
    root = Path(root)
    run_str = f"{run['name']}_g{run['gate']}"

    if (root / run_str).is_dir():
        return root / run_str

    old = root / run['name'] / run_str
    if old.is_dir():
        return old

    return None


def rsync_copy(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{src}/", f"{dest}/"],
        check=True,
    )


def ensure_run_downloaded(run, config):
    """Make sure this run's raw data exists under npx_directory, fetching it
    from a legacy server if necessary. No-op if already present."""
    npx_root = Path(config['npx_directory'])
    run_str = f"{run['name']}_g{run['gate']}"
    force = config.get('force_rerun_download', False)

    already_local = find_run_dir(run, npx_root) is not None
    if already_local:
        if not force:
            print(f"[{run_str}] already on scratch, skipping")
            return
        print(f"[{run_str}] already on scratch, but force_rerun_download is set, re-fetching...")

    dest = npx_root / run_str

    for legacy_root in config.get('legacy_raw_dirs', []):
        src = find_run_dir(run, legacy_root)
        if src:
            print(f"[{run_str}] found on {legacy_root}, copying to scratch...")
            rsync_copy(src, dest)
            return

    if already_local:
        print(f"[{run_str}] not found in legacy_raw_dirs, keeping existing local copy.")
        return

    raise FileNotFoundError(
        f"[{run_str}] not found on scratch or any legacy_raw_dirs."
    )


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)

    failed = []
    for run in config['runs']:
        try:
            ensure_run_downloaded(run, config)
        except Exception as exc:
            print(f"[ERROR] {run['name']}_g{run['gate']}: {exc}", file=sys.stderr)
            failed.append(run['name'])

    if failed:
        print(f"\nRaw data missing for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
