"""
Neuropixels curation pipeline.

Applies pretrained classification models to sorted data.
Must be run after pipeline_ks4.py (requires existing SortingAnalyzer).

Steps for each run/probe:
  1. UnitRefine -- classify units as neural/noise using a pretrained model

Usage:
  python bin/pipeline_curation.py config.yaml

Output (per probe, inside the existing ks4 folder):
  curation/
    unitrefine_labels.csv   -- unit IDs, predicted label, confidence score
"""

import sys
import yaml
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.curation as sc


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# --- Path helpers (mirror pipeline_ks4.py) ---

def ks4_dir(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    return Path(config['output_dir']) / f"catgt_{run_str}" / f"{run_str}_imec{prb}" / f"imec{prb}_ks4"


def analyzer_dir(run, prb, config):
    return ks4_dir(run, prb, config) / "analyzer"


def curation_dir(run, prb, config):
    return ks4_dir(run, prb, config) / "curation"


# --- Curation steps ---

def run_unitrefine(run, prb, config):
    ana_dir = analyzer_dir(run, prb, config)
    out_path = curation_dir(run, prb, config) / "unitrefine_labels.csv"

    if not ana_dir.exists():
        print(f"  [UnitRefine] No analyzer found for probe {prb}, skipping.")
        return

    if out_path.exists() and not config.get('force_rerun_curation'):
        print(f"  [UnitRefine] Labels exist for probe {prb}, skipping.")
        return

    print(f"  [UnitRefine] Running for probe {prb}...")
    analyzer = si.load_sorting_analyzer(ana_dir)

    labels = sc.auto_label_units(
        sorting_analyzer=analyzer,
        repo_id=config['curation']['unitrefine_model'],
        trusted=['numpy.dtype'],
    )

    curation_dir(run, prb, config).mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path)
    print(f"  [UnitRefine] Labels saved: {out_path}")

    counts = labels.iloc[:, 0].value_counts()
    for label, count in counts.items():
        print(f"    {label}: {count} units ({count / len(labels) * 100:.1f}%)")


def process_run(run, config):
    print(f"\n{'='*60}\nCuration: {run['name']}\n{'='*60}")
    for prb in run['probes']:
        run_unitrefine(run, prb, config)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    for run in config['runs']:
        process_run(run, config)


if __name__ == '__main__':
    main()