# Neuropixels Spike Sorting Pipeline

Automated pipeline for sorting Neuropixels recordings acquired with SpikeGLX.

**Steps:** CatGT → Kilosort 4 → SpikeInterface postprocessing → TPrime → UnitRefine curation

---

## Prerequisites

| Tool | Notes |
|------|-------|
| [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) | package manager |
| `si_ks4` micromamba environment | SpikeInterface + Kilosort 4 (see below) |
| `curation` micromamba environment | SpikeInterface + UnitRefine dependencies (see below) |
| CatGT | standalone binary, Linux version from https://billkarsh.github.io/SpikeGLX |
| TPrime | standalone binary, same site |
| `screen` | usually pre-installed; `sudo apt install screen` if missing |

### Setting up the `si_ks4` environment

```bash
micromamba create -n si_ks4 python=3.11
micromamba activate si_ks4
pip install spikeinterface[full,widgets]
pip install kilosort
```

### Setting up the `curation` environment

```bash
micromamba create -n curation python=3.11
micromamba activate curation
pip install uv

# UnitRefine — must be cloned (not on PyPI)
git clone https://github.com/anoushkajain/UnitRefine.git
uv pip install ./UnitRefine

# Bombcell (Python)
uv pip install bombcell
```

To launch the UnitRefine GUI for manual inspection:
```bash
uv run --directory UnitRefine unitrefine --project_folder my_project
```

---

## Setup

1. **Clone the repo** on your sorting machine:
   ```bash
   git clone <repo-url>
   cd SpikeSorting
   ```

2. **Copy the example config** and fill in your paths:
   ```bash
   cp config_example.yaml config1.yaml
   nano config1.yaml   # or any editor
   ```

3. **Set the paths** in your config:
   - `npx_directory` — raw SpikeGLX recordings (read-only is fine)
   - `catgt_read` — existing preprocessed CatGT data to read from
   - `catgt_write` — where new CatGT output is written (needs write access)
   - `output_dir` — where Kilosort and SpikeInterface results go
   - `catgt_bin` / `tprime_bin` — paths to the `runit.sh` scripts

4. **Add your recordings** under the `runs:` section of the config:
   ```yaml
   runs:
     - name: 240308_KK102   # SpikeGLX run name (before _gN)
       gate: 0
       triggers: "0,0"
       probes: [0]
   ```

---

## Running the pipeline

To run sorting and curation in one go:

```bash
bash bin/run_pipeline.sh config1.yaml
```

Or run the steps individually:

```bash
bash bin/run_sorting.sh config1.yaml    # sorting only
bash bin/run_curation.sh config1.yaml   # curation only (requires sorting to exist)
```

All scripts start a detached `screen` session named after the config file, so they keep running after you disconnect from SSH. Multiple configs can run in parallel — each gets its own session.

**Monitor progress:**
```bash
screen -r sorting_config1  # attach to the live session (Ctrl+A then D to detach again)
tail -f kk_ks4_YYMMDD.log  # or follow the log file
```

**Check running processes:**
```bash
screen -ls                 # list all screen sessions
ps aux | grep pipeline     # check if the process is alive
```

**Stop a session:**
```bash
screen -S sorting_config1 -X quit
```

You can also run the pipeline directly (without screen) if you don't need background execution:
```bash
micromamba run -n si_ks4 python bin/pipeline_ks4.py config1.yaml
```

---

## Running curation (UnitRefine)

Run this **after** sorting has completed. It classifies each unit as neural or noise using a pretrained model from HuggingFace.

```bash
bash bin/run_curation.sh config1.yaml
```

This starts in its own `screen` session (e.g. `curation_config1`).

**Monitor progress:**
```bash
screen -r curation_config1
tail -f logs/kk_curation_YYMMDD.log
```

Or run directly:
```bash
micromamba run -n curation python bin/pipeline_curation.py config1.yaml
```

---

## Output structure

```
output_dir/
  catgt_{run}_g{gate}/
    {run}_g{gate}_imec{prb}/
      imec{prb}_ks4/
        sorter_output/
          spike_times_sec.npy      # merged spike times in seconds
          spike_times_sec_adj.npy  # TPrime-aligned spike times
        analyzer/                  # SpikeInterface SortingAnalyzer
          labels.csv               # UnitRefine labels (Good/Noise); auto-detected by UnitRefine GUI
          extensions/
            quality_metrics/       # SNR, ISI violations, presence ratio, etc.
            template_metrics/      # waveform shape features (exp_decay, spread, etc.)
            ...
        bombcell/                  # Bombcell output (quality metrics, GUI data)
          unit_labels.csv          # unit IDs, Bombcell unit type (good/MUA/noise/non-somatic)
```

---

## Re-running specific steps

Set the relevant flag to `true` in your config and rerun:

| Flag | Effect |
|------|--------|
| `force_rerun_catgt: true` | Redo CatGT filtering even if output exists |
| `force_rerun_kilosort: true` | Redo sorting + all postprocessing from scratch |
| `force_rerun_metrics: true` | Recompute waveforms and metrics only (skip re-sorting) |
| `force_rerun_tprime: true` | Redo TPrime spike time alignment |
| `force_rerun_curation: true` | Redo UnitRefine classification even if labels exist |

---

## Tips

- **n_jobs**: Keep `n_jobs: 1` if data is on a network drive — parallel workers cause I/O contention and are slower than a single worker.
- **CatGT output**: CatGT logs to `CatGT.log` in the working directory, not to stdout. Check this file if CatGT appears to do nothing.
- **Multiple recordings**: Add multiple entries under `runs:` — the pipeline processes them in order.
- **Read-only catgt data**: If preprocessed CatGT data already exists on a read-only server, point `catgt_read` there. The pipeline will skip CatGT and read from it directly.