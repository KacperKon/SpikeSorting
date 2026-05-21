"""
Neuropixels spike sorting report generator.

Generates a PDF per probe with one page per sorted unit, combining:
  - Waveform templates on the 6 highest-amplitude channels (3×2 grid),
    with trough, peak, half-width, and recovery slope annotated on the best channel
  - ISI histogram
  - Spike amplitude over time (stability check)
  - Waveform metrics for cell type identification + quality metrics

Labels from Kilosort 4, UnitRefine (with confidence), and Bombcell are shown
in a colour-coded header.

Must be run after pipeline_curation.py.

Usage:
  python bin/pipeline_report.py config.yaml

Output (per probe):
  imec{prb}_ks4/report.pdf
"""

import sys
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import spikeinterface.full as si


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# --- Path helpers (mirror pipeline_ks4.py) ---

def ks4_dir(run, prb, config):
    run_str = f"{run['name']}_g{run['gate']}"
    return Path(config['output_dir']) / f"catgt_{run_str}" / f"{run_str}_imec{prb}" / f"imec{prb}_ks4"


def analyzer_dir(run, prb, config):
    return ks4_dir(run, prb, config) / "analyzer"


# --- Label loading ---

def _label_color(label):
    if label is None:
        return '#7f8c8d'
    s = str(label).lower()
    if s == 'sua' or 'good' in s or 'neural' in s:
        return '#27ae60'
    if 'mua' in s:
        return '#e67e22'
    if s in ('not-sua', 'not_sua') or 'noise' in s:
        return '#e74c3c'
    if 'non' in s or 'somatic' in s:
        return '#8e44ad'
    return '#7f8c8d'


def load_labels(run, prb, config):
    """Load unit labels from KS4, UnitRefine, and Bombcell."""
    ks_labels = {}
    ks_path = ks4_dir(run, prb, config) / 'sorter_output' / 'cluster_KSLabel.tsv'
    if ks_path.exists():
        df = pd.read_csv(ks_path, sep='\t')
        ks_labels = dict(zip(df['cluster_id'].astype(int), df['KSLabel']))

    ur_labels, ur_conf = {}, {}
    ur_path = analyzer_dir(run, prb, config) / 'labels.csv'
    if ur_path.exists():
        df = pd.read_csv(ur_path)
        ur_labels = dict(zip(df['unit_id'].astype(int), df['quality']))
        if 'confidence' in df.columns:
            ur_conf = dict(zip(df['unit_id'].astype(int), df['confidence']))

    bc_labels = {}
    bc_path = ks4_dir(run, prb, config) / 'bombcell' / 'unit_labels.csv'
    if bc_path.exists():
        df = pd.read_csv(bc_path)
        bc_labels = dict(zip(df['unit_id'].astype(int), df['label']))

    return ks_labels, ur_labels, ur_conf, bc_labels


# --- Per-unit figure ---

def _get_ext(analyzer, name):
    try:
        ext = analyzer.get_extension(name)
        return ext.get_data() if ext is not None else None
    except Exception as e:
        print(f"  [Report] Warning: failed to load extension '{name}': {e}")
        return None


def preload_ext_data(analyzer, bc_dir=None):
    """Load all extension data and per-unit amplitude arrays once before the plotting loop."""
    fs = analyzer.sorting.get_sampling_frequency()
    data = {
        'fs': fs,
        'templates':        _get_ext(analyzer, 'templates'),
        'quality_metrics':  _get_ext(analyzer, 'quality_metrics'),
        'template_metrics': _get_ext(analyzer, 'template_metrics'),
        'isi_histograms':   _get_ext(analyzer, 'isi_histograms'),
        'unit_times': {},
        'unit_amps':  {},
        'amp_error':  None,
        # Bombcell
        'bc_wf':         None,  # (n_units, n_channels, n_samples)
        'bc_ch':         None,  # (n_units,) best channel index per unit
        'bc_qm':         {},    # uid → row Series from _bc_qMetrics.csv
        'bc_peak_loc':   {},    # uid → sample index of main peak (for duration)
        'bc_trough_loc': {},    # uid → sample index of main trough (for duration)
    }

    if bc_dir is not None:
        try:
            import pandas as _pd
            bc_qm_df = _pd.read_csv(Path(bc_dir) / 'templates._bc_qMetrics.csv', index_col=0)
            for _, row in bc_qm_df.iterrows():
                data['bc_qm'][int(row['phy_clusterID'])] = row
            data['bc_wf'] = np.load(Path(bc_dir) / 'templates._bc_rawWaveforms.npy')
            data['bc_ch'] = np.load(Path(bc_dir) / 'templates._bc_rawWaveformPeakChannels.npy').astype(int)
            print(f"  [Report] Bombcell waveforms loaded: {data['bc_wf'].shape}")
            import pickle
            with open(Path(bc_dir) / 'for_GUI' / 'gui_data.pkl', 'rb') as _f:
                _gui = pickle.load(_f)
            data['bc_peak_loc']   = _gui.get('peak_loc_for_duration', {})
            data['bc_trough_loc'] = _gui.get('trough_loc_for_duration', {})
        except Exception as e:
            print(f"  [Report] Warning: failed to load Bombcell data: {e}")

    amp_ext = analyzer.get_extension('spike_amplitudes')
    if amp_ext is not None:
        try:
            raw = amp_ext.get_data()
            if isinstance(raw, (list, tuple)):
                all_amps = np.concatenate(raw)
            elif isinstance(raw, np.ndarray) and raw.ndim >= 1:
                all_amps = raw.ravel()
            else:
                all_amps = np.array(raw).ravel()
            sv = analyzer.sorting.to_spike_vector()
            for unit_idx, unit_id in enumerate(analyzer.unit_ids):
                mask = sv['unit_index'] == unit_idx
                times = sv['sample_index'][mask] / fs / 60
                amps  = np.abs(all_amps[mask])
                if len(times) > 5000:
                    idx = np.linspace(0, len(times) - 1, 5000, dtype=int)
                    times, amps = times[idx], amps[idx]
                data['unit_times'][unit_idx] = times
                data['unit_amps'][unit_idx]  = amps
        except Exception as e:
            data['amp_error'] = str(e)

    return data


def plot_unit_page(unit_id, unit_idx, ext_data, ks_labels, ur_labels, ur_conf, bc_labels):
    fig = plt.figure(figsize=(14, 10))
    fs  = ext_data['fs']
    uid = int(unit_id)

    # ── Colour-coded header ───────────────────────────────────────────
    ks_lbl = ks_labels.get(uid, 'N/A')
    ur_lbl = ur_labels.get(uid, 'N/A')
    ur_pct = ur_conf.get(uid, None)
    bc_lbl = bc_labels.get(uid, 'N/A')
    ur_str = f"UnitRefine: {ur_lbl}" + (f" ({ur_pct:.0%})" if ur_pct is not None else "")

    fig.text(0.03, 0.977, f"Unit {unit_id}", fontsize=13, fontweight='bold',
             va='top', color='#2c3e50')
    fig.text(0.22, 0.977, f"KS4: {ks_lbl}", fontsize=10, fontweight='bold',
             va='top', color=_label_color(ks_lbl))
    fig.text(0.38, 0.977, ur_str, fontsize=10, fontweight='bold',
             va='top', color=_label_color(ur_lbl))
    fig.text(0.70, 0.977, f"Bombcell: {bc_lbl}", fontsize=10, fontweight='bold',
             va='top', color=_label_color(bc_lbl))

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.92, bottom=0.08, left=0.07, right=0.97)

    # ── Waveforms 3×2 ────────────────────────────────────────────────
    # Prefer Bombcell waveforms (n_units, n_channels, n_samples); fall back to SI templates
    bc_wf_all = ext_data['bc_wf']
    bc_ch_all = ext_data['bc_ch']
    if bc_wf_all is not None and unit_idx < len(bc_wf_all):
        unit_tmpl = bc_wf_all[unit_idx].T          # → (n_samples, n_channels)
        best_ch   = int(bc_ch_all[unit_idx])
    else:
        unit_tmpl = ext_data['templates'][unit_idx] if ext_data['templates'] is not None else None
        best_ch   = None

    if unit_tmpl is not None:
        n_samples = unit_tmpl.shape[0]
        t_ms = np.arange(n_samples) / fs * 1000
        best_6 = np.argsort(np.ptp(unit_tmpl, axis=0))[::-1][:6]
        if best_ch is None:
            best_ch = best_6[0]
        # Ensure best_ch appears first in the 6-channel list
        best_6 = np.array([best_ch] + [c for c in best_6 if c != best_ch][:5])

        gs_wf = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0, 0],
                                        hspace=0.05, wspace=0.05)
        for i, ch_idx in enumerate(best_6):
            ax = fig.add_subplot(gs_wf[i // 2, i % 2])
            wf = unit_tmpl[:, ch_idx]
            ax.plot(t_ms, wf, color='#2c3e50', lw=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if ch_idx == best_ch:
                from matplotlib.lines import Line2D
                wf_source = 'Bombcell' if bc_wf_all is not None else 'SpikeInterface'

                # ── Scale bar (always shown) ──────────────────────────
                t_range   = t_ms[-1] - t_ms[0]
                a_range   = np.ptp(wf)
                bar_t     = 0.5
                bar_a     = next((v for v in [5,10,20,50,100,200,500]
                                  if v >= a_range * 0.2), 500)
                x0 = t_ms[0] + t_range * 0.05
                y0 = wf.min() - a_range * 0.08
                ax.hlines(y0, x0, x0 + bar_t, colors='k', lw=1.5, zorder=6, clip_on=False)
                ax.vlines(x0, y0, y0 + bar_a, colors='k', lw=1.5, zorder=6, clip_on=False)
                ax.text(x0 + bar_t / 2, y0 - a_range * 0.10,
                        f'{bar_t} ms', fontsize=5, ha='center', va='top', clip_on=False)
                ax.text(x0 - t_range * 0.03, y0 + bar_a / 2,
                        f'{bar_a} μV', fontsize=5, ha='right', va='center', clip_on=False)

                # ── Annotations: Bombcell positions only ─────────────
                bc_peak   = ext_data['bc_peak_loc'].get(uid)
                bc_trough = ext_data['bc_trough_loc'].get(uid)
                if bc_peak is not None and bc_trough is not None:
                    peak_i   = int(bc_peak)
                    trough_i = int(bc_trough)
                    t_val    = wf[trough_i]
                    p_val    = wf[peak_i]

                    ax.plot(t_ms[trough_i], t_val, 'o', color='#e74c3c', ms=5, zorder=5)
                    ax.plot(t_ms[peak_i],   p_val, 'o', color='#3498db', ms=5, zorder=5)

                    # Half-width bar at half trough depth
                    half = t_val / 2.0
                    crossings = np.where(np.diff((wf <= half).astype(int)))[0]
                    if len(crossings) >= 2:
                        ax.hlines(half, t_ms[crossings[0]], t_ms[crossings[1]],
                                  colors='#27ae60', lw=2, zorder=5)

                    # Trough-to-peak arrow
                    arrow_y = t_val * 1.20
                    ax.annotate('', xy=(t_ms[peak_i], arrow_y),
                                xytext=(t_ms[trough_i], arrow_y),
                                arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=1.2))

                    # Recovery slope tangent
                    gap = abs(peak_i - trough_i)
                    if gap > 10:
                        lo, hi = min(trough_i, peak_i), max(trough_i, peak_i)
                        mid = (lo + hi) // 2
                        sl  = slice(max(0, mid - gap // 4), mid + gap // 4)
                        if len(t_ms[sl]) > 2:
                            m, b = np.polyfit(t_ms[sl], wf[sl], 1)
                            x_ext = np.array([t_ms[lo], t_ms[hi]])
                            ax.plot(x_ext, m * x_ext + b, '--', color='#e67e22',
                                    lw=1.2, alpha=0.85, zorder=4)

                    legend_elems = [
                        Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', ms=4, label='trough'),
                        Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498db', ms=4, label='peak'),
                        Line2D([0],[0], color='#27ae60', lw=1.5, label='½-width'),
                        Line2D([0],[0], color='#8e44ad', lw=1.2, label='trough→peak'),
                        Line2D([0],[0], color='#e67e22', lw=1, ls='--', label='recovery'),
                    ]
                    ax.legend(handles=legend_elems, loc='upper right', fontsize=7,
                              framealpha=0.75, handlelength=1.4, borderpad=0.4,
                              labelspacing=0.2, handletextpad=0.5,
                              title=wf_source, title_fontsize=7)
                else:
                    ax.text(0.97, 0.96, wf_source, transform=ax.transAxes,
                            fontsize=5, va='top', ha='right', color='#7f8c8d')
    else:
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, 'Waveforms not available', ha='center', va='center',
                transform=ax.transAxes)

    # ── ISI histogram ────────────────────────────────────────────────
    ax_isi = fig.add_subplot(gs[0, 1])
    isi_data = ext_data['isi_histograms']
    if isi_data is not None:
        try:
            # SI returns (histograms, bins) tuple; bins in ms
            isi_hists, isi_bins = isi_data
            unit_isi = isi_hists[unit_idx]
            bw = np.diff(isi_bins)[0]
            # Mirror around 0 to show refractory period as gap in the middle
            n_half = min(int(np.ceil(50.0 / bw)), len(unit_isi))
            vals = unit_isi[:n_half]
            centers = isi_bins[:n_half] + bw / 2
            sym_centers = np.concatenate([-centers[::-1], centers])
            sym_vals = np.concatenate([vals[::-1], vals])
            ax_isi.bar(sym_centers, sym_vals, width=bw * 0.9,
                       color='#3498db', alpha=0.75, edgecolor='none')
            ax_isi.axvspan(-1.5, 1.5, color='#e74c3c', alpha=0.15, label='±1.5 ms')
            ax_isi.axvline(0, color='k', lw=0.5, alpha=0.4)
            ax_isi.set_xlim(-50, 50)
        except Exception as e:
            ax_isi.text(0.5, 0.5, f'ISI error:\n{e}', ha='center', va='center',
                        transform=ax_isi.transAxes, fontsize=7)
    else:
        ax_isi.text(0.5, 0.5, 'ISI not available', ha='center', va='center',
                    transform=ax_isi.transAxes)
    ax_isi.set_title('ISI Histogram', fontsize=10)
    ax_isi.set_xlabel('ISI (ms)', fontsize=9)
    ax_isi.set_ylabel('Count', fontsize=9)
    ax_isi.tick_params(labelsize=8)

    # ── Amplitude over time ──────────────────────────────────────────
    ax_amp = fig.add_subplot(gs[1, 0])
    if unit_idx in ext_data['unit_times']:
        u_times = ext_data['unit_times'][unit_idx]
        u_amps  = ext_data['unit_amps'][unit_idx]
        ax_amp.scatter(u_times, u_amps, s=3, alpha=0.5, color='#2c3e50',
                       rasterized=True, linewidths=0)
        # Binned average
        if len(u_times) >= 10:
            n_bins = min(50, len(u_times) // 5)
            edges = np.linspace(u_times[0], u_times[-1], n_bins + 1)
            bx = [(edges[i] + edges[i+1]) / 2 for i in range(n_bins)
                  if ((u_times >= edges[i]) & (u_times < edges[i+1])).sum() > 0]
            by = [u_amps[(u_times >= edges[i]) & (u_times < edges[i+1])].mean()
                  for i in range(n_bins)
                  if ((u_times >= edges[i]) & (u_times < edges[i+1])).sum() > 0]
            ax_amp.plot(bx, by, color='#e74c3c', lw=1.5, zorder=5)
        ax_amp.set_xlabel('Time (min)', fontsize=9)
        ax_amp.set_ylabel('Amplitude (μV)', fontsize=9)
    elif ext_data['amp_error']:
        ax_amp.text(0.5, 0.5, f'Error:\n{ext_data["amp_error"]}', ha='center', va='center',
                    transform=ax_amp.transAxes, fontsize=7)
    else:
        ax_amp.text(0.5, 0.5, 'Amplitudes not available', ha='center', va='center',
                    transform=ax_amp.transAxes)
    ax_amp.set_title('Amplitude over time', fontsize=10)
    ax_amp.tick_params(labelsize=8)

    # ── Metrics table ────────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[1, 1])
    ax_m.axis('off')

    qm = ext_data['quality_metrics']
    tm = ext_data['template_metrics']

    def val(df, col, fmt='.3f', mult=1.0):
        if df is None or uid not in df.index or col not in df.columns:
            return 'N/A'
        v = df.loc[uid, col]
        if isinstance(v, float) and np.isnan(v):
            return 'N/A'
        return f'{v * mult:{fmt}}'

    # Prefer Bombcell waveform metrics (samples → ms via /fs*1000); fall back to SI
    bc_row = ext_data['bc_qm'].get(uid)

    def bc_val(col, fmt, mult=1.0):
        if bc_row is None or col not in bc_row.index:
            return 'N/A'
        v = bc_row[col]
        if isinstance(v, float) and np.isnan(v):
            return 'N/A'
        return f'{v * mult:{fmt}}'

    cell_type_rows = [
        # (label, value_string)
        ('Trough-to-peak (ms)', bc_val('waveformDuration_peakTrough', '.3f', 1000.0 / fs)
                                if bc_row is not None
                                else val(tm, 'peak_to_trough_duration', '.3f', 1000.0)),
        ('Half-width (ms)',     bc_val('mainTrough_width', '.3f', 1000.0 / fs)
                                if bc_row is not None
                                else val(tm, 'trough_half_width', '.3f', 1000.0)),
        ('Peak/trough ratio',   bc_val('mainPeakToTroughRatio', '.3f')
                                if bc_row is not None
                                else val(tm, 'peak_after_to_trough_ratio', '.3f')),
        ('Recovery slope',      val(tm, 'recovery_slope', '.4f')),
    ]
    quality_rows = [
        ('firing_rate',          'Firing rate (Hz)',   qm, '.2f', 1.0),
        ('snr',                  'SNR',                qm, '.2f', 1.0),
        ('isi_violations_ratio', 'ISI violations (%)', qm, '.4f', 1.0),
        ('presence_ratio',       'Presence ratio',     qm, '.3f', 1.0),
    ]

    y = 0.97
    ax_m.text(0.0, y, '── Waveform ──', fontsize=9, fontweight='bold',
              transform=ax_m.transAxes, va='top', color='#2c3e50')
    y -= 0.12
    for label, value_str in cell_type_rows:
        ax_m.text(0.03, y, label, fontsize=8.5, transform=ax_m.transAxes, va='top')
        ax_m.text(0.85, y, value_str, fontsize=8.5, transform=ax_m.transAxes,
                  va='top', ha='right', fontweight='bold', color='#2980b9')
        y -= 0.11

    y -= 0.04
    ax_m.text(0.0, y, '── Quality ──', fontsize=9, fontweight='bold',
              transform=ax_m.transAxes, va='top', color='#2c3e50')
    y -= 0.12
    for col, label, df, fmt, mult in quality_rows:
        ax_m.text(0.03, y, label, fontsize=8.5, transform=ax_m.transAxes, va='top')
        ax_m.text(0.85, y, val(df, col, fmt, mult), fontsize=8.5, transform=ax_m.transAxes,
                  va='top', ha='right', fontweight='bold', color='#2980b9')
        y -= 0.11

    return fig


# --- Summary page ---

def plot_summary_page(run, prb, unit_ids, ks_labels, ur_labels, ur_conf, bc_labels, ks_dir):
    fig = plt.figure(figsize=(14, 10))
    run_str = f"{run['name']}_g{run['gate']}"
    fig.text(0.5, 0.96, f"{run_str}  —  probe {prb}", fontsize=16, fontweight='bold',
             ha='center', va='top', color='#2c3e50')
    fig.text(0.5, 0.91, f"{len(unit_ids)} units total", fontsize=12,
             ha='center', va='top', color='#7f8c8d')

    label_order = {'good': 0, 'Good': 0, 'neural': 0,
                   'mua': 1, 'MUA': 1, 'non-somatic': 2,
                   'noise': 3, 'Noise': 3}

    sources = [
        ('Kilosort 4',  {int(uid): ks_labels.get(int(uid), 'N/A') for uid in unit_ids}),
        ('UnitRefine',  {int(uid): ur_labels.get(int(uid), 'N/A') for uid in unit_ids}),
        ('Bombcell',    {int(uid): bc_labels.get(int(uid), 'N/A') for uid in unit_ids}),
    ]

    # Top row: three label-count bar charts
    ax_row = [fig.add_axes([0.07 + i * 0.31, 0.42, 0.24, 0.42]) for i in range(3)]
    for ax, (title, labels_dict) in zip(ax_row, sources):
        counts = {}
        for lbl in labels_dict.values():
            counts[str(lbl)] = counts.get(str(lbl), 0) + 1
        counts = dict(sorted(counts.items(), key=lambda x: (label_order.get(x[0], 4), x[0])))
        labels_list = list(counts.keys())
        values = list(counts.values())
        colors = [_label_color(l) for l in labels_list]
        bars = ax.barh(labels_list, values, color=colors, edgecolor='white', height=0.55)
        for bar, v in zip(bars, values):
            pct = v / len(unit_ids) * 100
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{v}  ({pct:.1f}%)', va='center', fontsize=9, color='#2c3e50')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8, color='#2c3e50')
        ax.set_xlabel('Units', fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max(values) * 1.45)

    # Bottom: KS4 drift over time
    ax_drift = fig.add_axes([0.07, 0.07, 0.88, 0.27])
    ops_path = ks_dir / 'sorter_output' / 'ops.npy'
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
        dshift = np.array(ops['dshift']).squeeze()   # drift per batch (µm)
        batch_size = int(ops.get('batch_size', ops.get('NT', 60000)))
        fs = float(ops['fs'])
        t_min = np.arange(len(dshift)) * batch_size / fs / 60
        ax_drift.plot(t_min, dshift, color='#2980b9', lw=1.2)
        ax_drift.axhline(0, color='#bdc3c7', ls='--', lw=0.8)
        ax_drift.set_xlabel('Time (min)', fontsize=9)
        ax_drift.set_ylabel('Drift (µm)', fontsize=9)
        ax_drift.set_title('Probe drift (Kilosort 4 estimate)', fontsize=10)
        ax_drift.tick_params(labelsize=8)
        ax_drift.spines['top'].set_visible(False)
        ax_drift.spines['right'].set_visible(False)
    except Exception as e:
        ax_drift.text(0.5, 0.5, f'Drift not available:\n{e}', ha='center', va='center',
                      transform=ax_drift.transAxes, fontsize=8)
        ax_drift.set_title('Probe drift (Kilosort 4 estimate)', fontsize=10)

    return fig


# --- Unit ordering ---

def _is_good_label(label):
    """Return True for labels that represent well-isolated single units."""
    s = str(label).lower()
    return s in ('sua', 'good', 'neural') or 'good' in s or 'neural' in s


def _sort_units(unit_ids, ur_labels, ur_conf):
    """Sort units from best to worst:
    Single-unit / good labels descending by confidence, then everything else
    ascending by confidence (least confident = most borderline first).
    """
    def key(uid):
        uid_int = int(uid)
        label = ur_labels.get(uid_int, 'N/A')
        conf = ur_conf.get(uid_int, 0.5)
        if _is_good_label(label):
            return (0, -conf)
        else:
            return (1, conf)
    return sorted(unit_ids, key=key)


# --- Report generation ---

def generate_report(run, prb, config):
    ana_dir = analyzer_dir(run, prb, config)
    run_str = f"{run['name']}_g{run['gate']}_imec{prb}"
    out_path = ks4_dir(run, prb, config) / f"{run_str}_report.pdf"

    if not ana_dir.exists():
        print(f"  [Report] No analyzer for probe {prb}, skipping.")
        return

    if out_path.exists() and not config.get('force_rerun_report'):
        print(f"  [Report] Report exists for probe {prb}, skipping.")
        return

    print(f"  [{_ts()}] [Report] Generating for probe {prb}...")
    analyzer = si.load_sorting_analyzer(ana_dir)
    ks_labels, ur_labels, ur_conf, bc_labels = load_labels(run, prb, config)

    unit_ids = analyzer.unit_ids
    unit_id_to_idx = {uid: i for i, uid in enumerate(unit_ids)}
    sorted_ids = _sort_units(unit_ids, ur_labels, ur_conf)

    print(f"  [{_ts()}] [Report] Pre-loading extensions...")
    bc_dir = ks4_dir(run, prb, config) / 'bombcell'
    ext_data = preload_ext_data(analyzer, bc_dir=bc_dir if bc_dir.exists() else None)

    n_failed = 0
    with PdfPages(out_path) as pdf:
        summary = plot_summary_page(run, prb, unit_ids, ks_labels, ur_labels, ur_conf, bc_labels,
                                    ks4_dir(run, prb, config))
        pdf.savefig(summary, dpi=200)
        plt.close(summary)

        for unit_id in sorted_ids:
            unit_idx = unit_id_to_idx[unit_id]
            try:
                fig = plot_unit_page(unit_id, unit_idx, ext_data,
                                     ks_labels, ur_labels, ur_conf, bc_labels)
                pdf.savefig(fig, dpi=200)
                plt.close(fig)
            except Exception as e:
                print(f"    [Report] Warning: unit {unit_id} failed: {e}")
                plt.close('all')
                n_failed += 1

    msg = f"  [{_ts()}] [Report] Saved: {out_path} ({len(unit_ids)} units"
    if n_failed:
        msg += f", {n_failed} failed"
    print(msg + ")")


def process_run(run, config):
    print(f"\n{'='*60}\n[{_ts()}] Report: {run['name']}\n{'='*60}")
    for prb in run['probes']:
        generate_report(run, prb, config)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    config = load_config(config_path)
    for run in config['runs']:
        process_run(run, config)


if __name__ == '__main__':
    main()
