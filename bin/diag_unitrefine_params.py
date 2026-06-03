"""
Compare stored SortingAnalyzer metric parameters against UnitRefine model expectations.

Usage:
    micromamba run -n si_ks4 python bin/diag_unitrefine_params.py <analyzer_path>

Example:
    micromamba run -n si_ks4 python bin/diag_unitrefine_params.py \
        /mnt/raidstorage/sort/catgt_230621_KK056_g0/230621_KK056_g0_imec0/imec0_ks4/analyzer
"""

import sys
import json
import urllib.request
from pathlib import Path
import spikeinterface.full as si

MODEL_INFO_URL = (
    "https://huggingface.co/AnoushkaJain3/UnitRefine-mice-sua-classifier"
    "/raw/main/model_info.json"
)

def fetch_model_info():
    with urllib.request.urlopen(MODEL_INFO_URL) as r:
        return json.loads(r.read())

def get_ext_params(analyzer, ext_name):
    ext = analyzer.get_extension(ext_name)
    if ext is None:
        return None
    return ext.params

def compare(label, model_val, stored_val):
    match = model_val == stored_val
    status = "OK" if match else "MISMATCH"
    print(f"  [{status}] {label}")
    if not match:
        print(f"           model  : {model_val}")
        print(f"           stored : {stored_val}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ana_path = Path(sys.argv[1])
    print(f"Loading analyzer: {ana_path}")
    analyzer = si.load_sorting_analyzer(ana_path)

    print("\nFetching UnitRefine model_info.json...")
    try:
        model_info = fetch_model_info()
    except Exception as e:
        print(f"  Could not fetch model_info.json: {e}")
        model_info = None

    # --- Quality metrics ---
    print("\n=== quality_metrics ===")
    qm_params = get_ext_params(analyzer, 'quality_metrics')
    if qm_params is None:
        print("  NOT COMPUTED in this analyzer")
    else:
        print("  Stored params:")
        for k, v in qm_params.items():
            print(f"    {k}: {v}")
        if model_info:
            model_qm = model_info.get('metric_params', {}).get('quality_metric_params', {})
            model_names = set(model_qm.get('metric_names', []))
            stored_names = set(qm_params.get('metric_names') or [])
            missing = model_names - stored_names
            extra   = stored_names - model_names
            if missing:
                print(f"\n  MISSING metrics (model expects but not computed): {sorted(missing)}")
            if extra:
                print(f"  EXTRA metrics (computed but model doesn't use):   {sorted(extra)}")
            if not missing and not extra:
                print("  Metric names match exactly.")

    # --- Template metrics ---
    print("\n=== template_metrics ===")
    tm_params = get_ext_params(analyzer, 'template_metrics')
    if tm_params is None:
        print("  NOT COMPUTED in this analyzer")
    else:
        print("  Stored params:")
        for k, v in tm_params.items():
            print(f"    {k}: {v}")
        if model_info:
            model_tm = model_info.get('metric_params', {}).get('template_metric_params', {})
            model_names = set(model_tm.get('metric_names', []))
            stored_names = set(tm_params.get('metric_names') or [])
            missing = model_names - stored_names
            extra   = stored_names - model_names
            if missing:
                print(f"\n  MISSING metrics (model expects but not computed): {sorted(missing)}")
            if extra:
                print(f"  EXTRA metrics (computed but model doesn't use):   {sorted(extra)}")
            if not missing and not extra:
                print("  Metric names match exactly.")

    print()

if __name__ == '__main__':
    main()