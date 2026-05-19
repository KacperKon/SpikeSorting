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
import subprocess
import numpy as np
import yaml
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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
    Checks catgt_read first; falls back to catgt_write if not found there.
    CatGT always writes to catgt_write.
    """
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
        print(f"  [CatGT] Running: {' '.join(str(c) for c in cmd)}")
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

    if not config.get('force_rerun_kilosort') and (output_dir / 'spikeinterface_log.json').exists():
        print(f"  [KS4] Output exists for probe {prb}, loading.")
        return si.load_extractor(output_dir), recording

    print(f"  [KS4] Running Kilosort 4 for probe {prb}...")
    sorting = ss.run_sorter(
        sorter_name='kilosort4',
        recording=recording,
        folder=output_dir,
        verbose=True,
        remove_existing_folder=config.get('force_rerun_kilosort', False),
        **config.get('kilosort4_params', {}),
    )
    return sorting, recording


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

    if config.get('force_rerun_metrics') and not config.get('force_rerun_kilosort') and ana_dir.exists():
        print(f"  [Postprocessing] Recomputing metrics for probe {prb}...")
        si.set_global_job_kwargs(n_jobs=config.get('n_jobs', 4), chunk_duration='1s')
        analyzer = si.load_sorting_analyzer(ana_dir)
        analyzer.set_temporary_recording(recording)
        analyzer.compute([
            'random_spikes',
            'waveforms',
            'noise_levels',
            'templates',
            'spike_amplitudes',
            'spike_locations',
            'unit_locations',
            'principal_components',
            'correlograms',
            'isi_histograms'
        ])
        analyzer.compute('template_metrics', include_multi_channel_metrics=True)
        analyzer.compute('quality_metrics', qm_params={'drift': config.get('quality_metrics', {}).get('drift', {})})
        return analyzer

    print(f"  [Postprocessing] Computing waveforms and quality metrics for probe {prb}...")
    si.set_global_job_kwargs(n_jobs=config.get('n_jobs', 4), chunk_duration='1s')

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format='binary_folder',
        folder=ana_dir,
        overwrite=True,
    )
    analyzer.compute([
        'random_spikes',
        'waveforms',
        'noise_levels',
        'templates',
        'spike_amplitudes',
        'spike_locations',
        'unit_locations',
        'principal_components',
        'correlograms',
        'isi_histograms',
    ])
    analyzer.compute('template_metrics', include_multi_channel_metrics=True)
    analyzer.compute('quality_metrics', qm_params={'drift': config.get('quality_metrics', {}).get('drift', {})})
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

    Sync edge files are read from whichever of catgt_read / catgt_write holds the data.
    Spike times (input and output) live under output_dir.
    """
    run_str = f"{run['name']}_g{run['gate']}"
    tprime_cfg = config['tprime']
    tostream = catgt_run_dir(run, config) / tprime_cfg['tostream_file'].format(run=run_str)

    for prb in run['probes']:
        fromstream = catgt_prb_dir(run, prb, config) / tprime_cfg['fromstream_file'].format(run=run_str, prb=prb)
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
        print(f"  [TPrime] Running: {' '.join(str(c) for c in cmd)}")
        subprocess.check_call([str(c) for c in cmd])


def process_run(run, config):
    print(f"\n{'='*60}\nProcessing: {run['name']}\n{'='*60}")
    run_catgt(run, config)
    for prb in run['probes']:
        sorting, recording = load_or_run_kilosort4(run, prb, config)
        load_or_run_postprocessing(sorting, recording, run, prb, config)
        export_spike_times_for_tprime(sorting, recording, run, prb, config)
    run_tprime(run, config)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    for run in config['runs']:
        process_run(run, config)


if __name__ == '__main__':
    main()