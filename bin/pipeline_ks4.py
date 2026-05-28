"""
Neuropixels spike sorting pipeline.

Steps for each run/probe:
  1. CatGT          -- concatenate triggers, filter AP/LFP, extract sync events
  2. Kilosort 4     -- spike detection and clustering
  3. Postprocessing -- waveforms, templates, quality metrics (via SpikeInterface)
  4. TPrime         -- align spike times to a common reference clock

Usage:
  python pipeline.py config.yaml

Directory roots (configured in config.yaml):
  npx_directory : raw SpikeGLX recordings
  catgt_read    : existing preprocessed recordings to read from (may be read-only)
  catgt_write   : where new CatGT output is written (must be writable)
  output_dir    : Kilosort 4 results and SpikeInterface analyzer
"""

import sys
import shutil
import subprocess
import multiprocessing
import traceback
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# --- Path helpers ---

def raw_recording_parent(run, config):
    """
    Returns the directory to pass as CatGT's -dir flag.

    Supports two raw data layouts automatically:
      New: {npx_directory}/{name}_g{gate}/          (recording folder directly under npx_directory)
      Old: {npx_directory}/{name}/{name}_g{gate}/   (wrapped in an experiment sub-folder)
    """
    root = Path(config['npx_directory'])
    run_str = f"{run['name']}_g{run['gate']}"

    if (root / run_str).is_dir():
        return root

    old_parent = root / run['name']
    if (old_parent / run_str).is_dir():
        return old_parent

    raise FileNotFoundError(
        f"Cannot find recording '{run_str}'.\n"
        f"  Tried new layout: {root / run_str}\n"
        f"  Tried old layout: {old_parent / run_str}"
    )


def _catgt_root(run, config):
    """
    Resolves which catgt directory holds the preprocessed data for a run.
    If copy_raw_to_local is enabled and the local copy exists, prefer it.
    Otherwise checks catgt_read first; falls back to catgt_write.
    CatGT always writes to catgt_write.
    """
    run_str = f"{run['name']}_g{run['gate']}"
    if config.get('copy_raw_to_local') and config.get('catgt_local'):
        if (Path(config['catgt_local']) / f"catgt_{run_str}").exists():
            return Path(config['catgt_local'])
    if (Path(config['catgt_read']) / f"catgt_{run_str}").exists():
        return Path(config['catgt_read'])
    return Path(config['catgt_write'])


def _catgt_sync_root(run, config):
    # Like _catgt_root but never returns catgt_local — sync .txt files are not
    # copied there, only the .bin and .meta files are.
    run_str = f"{run['name']}_g{run['gate']}"
    if (Path(config['catgt_read']) / f"catgt_{run_str}").exists():
        return Path(config['catgt_read'])
    return Path(config['catgt_write'])


def catgt_run_dir(run, config):
    run_str = f"{run['name']}_g{run['gate']}"
    return _catgt_root(run, config) / f"catgt_{run_str}"


def catgt_prb_dir(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    return catgt_run_dir(run, config) / f"{run_str}_imec{prb}"


def catgt_bin_path(run, prb, config):
    # The concatenated AP binary written by CatGT; used as input to Kilosort.
    run_str = f"{run['name']}_g{run['gate']}"
    return catgt_prb_dir(run, prb, config) / f"{run_str}_tcat.imec{prb}.ap.bin"


def ks4_dir(run, prb, config):
    # Mirrors the catgt sub-folder structure under output_dir for easy cross-referencing.
    run_str = f"{run['name']}_g{run['gate']}"
    return Path(config['output_dir']) / f"catgt_{run_str}" / f"{run_str}_imec{prb}" / f"imec{prb}_ks4"


def analyzer_dir(run, prb, config):
    return ks4_dir(run, prb, config) / "analyzer"


# --- Pipeline steps ---

def run_catgt(run, config):
    """
    Run CatGT to concatenate triggers, filter AP/LFP bands, and extract sync pulses.

    Skip condition: output .bin already exists in catgt_read or catgt_write
                    and force_rerun_catgt is false.
    New output is always written to catgt_write.
    NI streams are extracted only on the first probe to avoid duplicates.
    """
    run_str = f"{run['name']}_g{run['gate']}"
    for i, prb in enumerate(run['probes']):
        if not config.get('force_rerun_catgt') and catgt_bin_path(run, prb, config).exists():
            print(f"  [CatGT] Output exists for {run_str} probe {prb}, skipping.")
            continue

        # Include NI extraction flags only for the first probe.
        stream_str = '-ap -ni' if i == 0 else '-ap'
        cmd = [
            config['catgt_bin'],
            f"-dir={raw_recording_parent(run, config)}",
            f"-run={run['name']}",
            f"-g={run['gate']}",
            f"-t={run['triggers']}",
            f"-prb={prb}",
            *stream_str.split(),
            *config['catgt_cmd_string'].split(),
            *config['ni_extract_string'].split(),
            f"-dest={config['catgt_write']}",
        ]
        print(f"  [{_ts()}] [CatGT] Running: {' '.join(str(c) for c in cmd)}")
        subprocess.check_call([str(c) for c in cmd])


def load_or_run_kilosort4(run, prb, config):
    """
    Run Kilosort 4 on one probe, or load existing results if already sorted.

    Reads the CatGT-filtered recording from whichever of catgt_read / catgt_write
    holds the data. Kilosort output is written to output_dir.

    Returns (sorting, recording) -- both are SpikeInterface objects.
    """
    output_dir = ks4_dir(run, prb, config)
    bin_path = catgt_bin_path(run, prb, config)
    recording = se.read_spikeglx(bin_path.parent, stream_id=f"imec{prb}.ap")

    log_exists = (output_dir / 'spikeinterface_log.json').exists()
    if not config.get('force_rerun_kilosort') and log_exists:
        print(f"  [KS4] Output exists for probe {prb}, loading.")
        # Load sorting from the SortingAnalyzer when it exists, not from the raw
        # sorter folder. Downstream tools (Bombcell) write TSV files into
        # sorter_output/ with float cluster IDs, which cause SI to fail on reload.
        # The analyzer stores the sorting with correct integer IDs.
        ana_dir = analyzer_dir(run, prb, config)
        if ana_dir.exists():
            return si.load_sorting_analyzer(ana_dir).sorting, recording
        return si.load(output_dir), recording

    # Remove an incomplete folder (exists but no log file) so KS4 can start fresh.
    remove_existing = config.get('force_rerun_kilosort', False) or (output_dir.exists() and not log_exists)
    if remove_existing and output_dir.exists() and not config.get('force_rerun_kilosort'):
        print(f"  [KS4] Incomplete output found for probe {prb}, removing and re-sorting.")

    print(f"  [{_ts()}] [KS4] Running Kilosort 4 for probe {prb}...")
    sorting = ss.run_sorter(
        sorter_name='kilosort4',
        recording=recording,
        folder=output_dir,
        verbose=True,
        remove_existing_folder=remove_existing,
        **config.get('kilosort4_params', {}),
    )
    return sorting, recording


def _run_si_extensions(analyzer, max_spikes_per_unit, config, prb):
    """Run all SpikeInterface extensions with per-step logging."""
    steps = [
        ('random_spikes',       lambda: analyzer.compute('random_spikes', max_spikes_per_unit=max_spikes_per_unit)),
        ('waveforms',           lambda: analyzer.compute('waveforms')),
        ('noise_levels',        lambda: analyzer.compute('noise_levels')),
        ('templates',           lambda: analyzer.compute('templates')),
        ('spike_amplitudes',    lambda: analyzer.compute('spike_amplitudes')),
        ('spike_locations',     lambda: analyzer.compute('spike_locations')),
        ('unit_locations',      lambda: analyzer.compute('unit_locations')),
        ('principal_components',lambda: analyzer.compute('principal_components')),
        ('correlograms',        lambda: analyzer.compute('correlograms')),
        ('isi_histograms',      lambda: analyzer.compute('isi_histograms')),
        ('template_metrics',    lambda: analyzer.compute('template_metrics', include_multi_channel_metrics=True)),
        ('quality_metrics',     lambda: analyzer.compute('quality_metrics',
                                    metric_params={'drift': config.get('quality_metrics', {}).get('drift', {})})),
    ]
    for name, fn in steps:
        print(f"  [{_ts()}] [SI prb{prb}] {name}...", flush=True)
        fn()
        print(f"  [{_ts()}] [SI prb{prb}] {name} done.", flush=True)


def load_or_run_postprocessing(sorting, recording, run, prb, config):
    """
    Build a SpikeInterface SortingAnalyzer with waveforms and quality metrics.

    Computed extensions (stored in the analyzer folder under output_dir):
      - random_spikes / waveforms / templates : spike shapes
      - noise_levels                           : per-channel noise estimate
      - spike_amplitudes                       : per-spike amplitude
      - spike_locations / unit_locations       : estimated spike/unit depth along probe
      - principal_components                   : PCA projections
      - correlograms / isi_histograms          : refractory period checks
      - template_metrics                       : waveform shape features (peak_trough_ratio,
                                                 half_width, exp_decay, spread, etc.)
      - quality_metrics                        : SNR, ISI violations, presence ratio, drift, etc.

    Skip condition: analyzer folder already exists and force_rerun_kilosort is false.
    """
    ana_dir = analyzer_dir(run, prb, config)

    if not config.get('force_rerun_kilosort') and not config.get('force_rerun_metrics') and ana_dir.exists():
        print(f"  [Postprocessing] Analyzer exists for probe {prb}, loading.")
        return si.load_sorting_analyzer(ana_dir)

    chunk_duration = config.get('chunk_duration', '1s')
    max_spikes_per_unit = config.get('max_spikes_per_unit', 500)

    if config.get('force_rerun_metrics') and not config.get('force_rerun_kilosort') and ana_dir.exists():
        print(f"  [{_ts()}] [Postprocessing] Recomputing metrics for probe {prb}...")
        si.set_global_job_kwargs(n_jobs=config.get('n_jobs', 4), chunk_duration=chunk_duration)
        analyzer = si.load_sorting_analyzer(ana_dir)
        analyzer.set_temporary_recording(recording)
        _run_si_extensions(analyzer, max_spikes_per_unit, config, prb)
        return analyzer

    print(f"  [{_ts()}] [Postprocessing] Starting SpikeInterface postprocessing for probe {prb}...")
    si.set_global_job_kwargs(n_jobs=config.get('n_jobs', 4), chunk_duration=chunk_duration)

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format='binary_folder',
        folder=ana_dir,
        overwrite=True,
    )
    _run_si_extensions(analyzer, max_spikes_per_unit, config, prb)
    return analyzer


def export_spike_times_for_tprime(sorting, recording, run, prb, config):
    """
    Save all spike times (in seconds) to a .npy file for TPrime alignment.

    Spikes from all units are merged and sorted chronologically, which is the
    format expected by TPrime's -events flag.
    """
    out_path = ks4_dir(run, prb, config) / 'sorter_output' / 'spike_times_sec.npy'
    if out_path.exists() and not config.get('force_rerun_kilosort') and not config.get('force_rerun_tprime'):
        return

    fs = recording.get_sampling_frequency()
    all_spikes = np.sort(np.concatenate([
        sorting.get_unit_spike_train(uid, segment_index=0)
        for uid in sorting.unit_ids
    ])) / fs
    np.save(out_path, all_spikes)
    print(f"  [TPrime prep] Spike times saved: {out_path}")


def run_tprime(run, config):
    """
    Run TPrime to remap spike times from each probe's clock to the NI (reference) clock.

    Sync edge files are read from catgt_read / catgt_write (never catgt_local, as only
    the .bin and .meta are copied there, not the sync .txt files).
    Spike times (input and output) live under output_dir.
    """
    run_str = f"{run['name']}_g{run['gate']}"
    tprime_cfg = config['tprime']
    sync_root = _catgt_sync_root(run, config)
    tostream = sync_root / f"catgt_{run_str}" / tprime_cfg['tostream_file'].format(run=run_str)

    for prb in run['probes']:
        fromstream = sync_root / f"catgt_{run_str}" / f"{run_str}_imec{prb}" / tprime_cfg['fromstream_file'].format(run=run_str, prb=prb)
        spike_in  = ks4_dir(run, prb, config) / 'sorter_output' / 'spike_times_sec.npy'
        spike_out = ks4_dir(run, prb, config) / 'sorter_output' / 'spike_times_sec_adj.npy'

        if spike_out.exists() and not config.get('force_rerun_tprime'):
            print(f"  [TPrime] Output exists for probe {prb}, skipping.")
            continue

        cmd = [
            config['tprime_bin'],
            f"-syncperiod={tprime_cfg['sync_period']}",
            f"-tostream={tostream}",
            f"-fromstream=0,{fromstream}",
            f"-events=0,{spike_in},{spike_out}",
        ]
        print(f"  [{_ts()}] [TPrime] Running: {' '.join(str(c) for c in cmd)}")
        subprocess.check_call([str(c) for c in cmd])


def copy_bin_to_local(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    src_root = None
    for candidate in [config['catgt_read'], config['catgt_write']]:
        if (Path(candidate) / f"catgt_{run_str}").exists():
            src_root = Path(candidate)
            break
    if src_root is None:
        print(f"  [LocalCopy] Source not found for probe {prb}, skipping.")
        return

    prb_subdir = f"catgt_{run_str}/{run_str}_imec{prb}"
    src_dir = src_root / prb_subdir
    dst_dir = Path(config['catgt_local']) / prb_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ['.ap.bin', '.ap.meta']:
        src = src_dir / f"{run_str}_tcat.imec{prb}{suffix}"
        dst = dst_dir / src.name
        if dst.exists():
            print(f"  [LocalCopy] Already local: {dst.name}, skipping.")
            continue
        if not src.exists():
            print(f"  [LocalCopy] Source not found: {src.name}, skipping.")
            continue
        print(f"  [LocalCopy] Copying {src.name} to {dst_dir}...")
        shutil.copy2(src, dst)
        print(f"  [LocalCopy] Done.")


def clear_local_bin(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    prb_dir = Path(config['catgt_local']) / f"catgt_{run_str}" / f"{run_str}_imec{prb}"
    for suffix in ['.ap.bin', '.ap.meta']:
        f = prb_dir / f"{run_str}_tcat.imec{prb}{suffix}"
        if f.exists():
            f.unlink()
            print(f"  [LocalCopy] Deleted {f.name}")


def process_run(run, config):
    print(f"\n{'='*60}\n[{_ts()}] Processing: {run['name']}\n{'='*60}")
    if config.get('copy_raw_to_local'):
        for prb in run['probes']:
            copy_bin_to_local(run, prb, config)
    run_catgt(run, config)
    for prb in run['probes']:
        sorting, recording = load_or_run_kilosort4(run, prb, config)
        load_or_run_postprocessing(sorting, recording, run, prb, config)
        export_spike_times_for_tprime(sorting, recording, run, prb, config)
    run_tprime(run, config)
    if config.get('copy_raw_to_local') and config.get('clear_local_copy'):
        for prb in run['probes']:
            clear_local_bin(run, prb, config)


def _process_run_worker(run, config):
    """Worker executed in a child process for one run."""
    try:
        process_run(run, config)
    except Exception:
        print(f"\n[ERROR] Run {run['name']} failed:")
        traceback.print_exc()
        sys.exit(1)


def main():
    multiprocessing.set_start_method('spawn', force=True)
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    if config.get('copy_raw_to_local') and not config.get('catgt_local'):
        raise ValueError("copy_raw_to_local is true but catgt_local is not set in config.")

    timeout_s = config.get('run_timeout_hours', 12) * 3600
    failed = []

    for run in config['runs']:
        p = multiprocessing.Process(target=_process_run_worker, args=(run, config))
        p.start()
        p.join(timeout=timeout_s)

        if p.is_alive():
            print(f"\n[{_ts()}] TIMEOUT ({config.get('run_timeout_hours', 12)}h) — "
                  f"killing {run['name']} and moving on.")
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
            failed.append((run['name'], 'timeout'))
        elif p.exitcode != 0:
            failed.append((run['name'], f'exit code {p.exitcode}'))

    if failed:
        print(f"\n[{_ts()}] Runs that did not complete:")
        for name, reason in failed:
            print(f"  {name}: {reason}")
        sys.exit(1)


if __name__ == '__main__':
    main()