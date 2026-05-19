"""
Neuropixels curation pipeline.

Applies pretrained classification models to sorted data.
Must be run after pipeline_ks4.py (requires existing SortingAnalyzer).

Steps for each run/probe:
  1. UnitRefine  -- classify units as neural/noise using a pretrained model
  2. Bombcell    -- classify units as good/MUA/noise/non-somatic

Usage:
  python bin/pipeline_curation.py config.yaml

Output (per probe, inside the existing ks4 folder):
  curation/
    unitrefine_labels.csv   -- unit IDs, predicted label, confidence score
    bombcell_labels.csv     -- unit IDs, Bombcell unit type
    bombcell/               -- full Bombcell output (quality metrics, GUI data)
"""

import sys
import yaml
from pathlib import Path

import pandas as pd
import spikeinterface.full as si
import spikeinterface.curation as sc
import bombcell as bc


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


def _catgt_root(run, config):
    run_str = f"{run['name']}_g{run['gate']}"
    if (Path(config['catgt_read']) / f"catgt_{run_str}").exists():
        return Path(config['catgt_read'])
    return Path(config['catgt_write'])


def catgt_bin_path(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    prb_dir = _catgt_root(run, config) / f"catgt_{run_str}" / f"{run_str}_imec{prb}"
    return prb_dir / f"{run_str}_tcat.imec{prb}.ap.bin"


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

    labels = sc.model_based_label_units(
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


def run_bombcell(run, prb, config):
    ks_dir = ks4_dir(run, prb, config) / 'sorter_output'
    bin_path = catgt_bin_path(run, prb, config)
    meta_path = bin_path.with_suffix('.meta')
    save_dir = curation_dir(run, prb, config) / 'bombcell'
    out_path = curation_dir(run, prb, config) / 'bombcell_labels.csv'

    if not ks_dir.exists():
        print(f"  [Bombcell] No KS4 output found for probe {prb}, skipping.")
        return

    if out_path.exists() and not config.get('force_rerun_curation'):
        print(f"  [Bombcell] Labels exist for probe {prb}, skipping.")
        return

    print(f"  [Bombcell] Running for probe {prb}...")

    param = bc.get_default_parameters(
        kilosort_path=str(ks_dir),
        raw_file=str(bin_path),
        kilosort_version=4,
        meta_file=str(meta_path),
    )
    for k, v in config.get('bombcell', {}).items():
        param[k] = v

    save_dir.mkdir(parents=True, exist_ok=True)
    quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
        str(ks_dir), str(save_dir), param
    )

    labels_df = pd.DataFrame({
        'unit_id': range(len(unit_type_string)),
        'label': unit_type_string,
    })
    labels_df.to_csv(out_path, index=False)
    print(f"  [Bombcell] Labels saved: {out_path}")

    counts = labels_df['label'].value_counts()
    for label, count in counts.items():
        print(f"    {label}: {count} units ({count / len(labels_df) * 100:.1f}%)")


def process_run(run, config):
    print(f"\n{'='*60}\nCuration: {run['name']}\n{'='*60}")
    for prb in run['probes']:
        run_unitrefine(run, prb, config)
        run_bombcell(run, prb, config)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    for run in config['runs']:
        process_run(run, config)


if __name__ == '__main__':
    main()
