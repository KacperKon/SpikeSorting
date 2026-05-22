import numpy as np, pickle
from pathlib import Path

bc_dir = Path(r'C:\Data\catgt_230814_KK060_g0\230814_KK060_g0_imec0\imec0_ks4\bombcell')

wf_all = np.load(bc_dir / 'templates._bc_rawWaveforms.npy')   # (171, 384, 61)
ch_all = np.load(bc_dir / 'templates._bc_rawWaveformPeakChannels.npy').astype(int)

with open(bc_dir / 'for_GUI' / 'gui_data.pkl', 'rb') as f:
    gui = pickle.load(f)
pl = gui['peak_loc_for_duration']
tl = gui['trough_loc_for_duration']

print(f'wf shape: {wf_all.shape}   ch range: {ch_all.min()}-{ch_all.max()}')
print()
print('uid   bc_ch  stored_tr  actual_tr(ch)  actual_tr(ch-1)  stored_pk  actual_pk(ch)  actual_pk(ch-1)  trough_ok?')
print('-' * 110)

wrong = []
for uid in range(min(30, len(wf_all))):
    ch = int(ch_all[uid])
    wf_ch  = wf_all[uid, ch, :]
    wf_ch1 = wf_all[uid, max(ch-1, 0), :]

    atr_ch  = int(np.argmin(wf_ch))
    apk_ch  = int(np.argmax(wf_ch))
    atr_ch1 = int(np.argmin(wf_ch1))
    apk_ch1 = int(np.argmax(wf_ch1))

    s_tr = int(tl[uid])
    s_pk = int(pl[uid])
    ok = 'OK' if abs(s_tr - atr_ch) <= 1 else 'MISMATCH'
    ok1 = 'OK' if abs(s_tr - atr_ch1) <= 1 else ''
    print(f'{uid:3d} {ch:7d} {s_tr:10d} {atr_ch:14d} {atr_ch1:16d} {s_pk:10d} {apk_ch:14d} {apk_ch1:16d}  {ok} {ok1}')
    if abs(s_tr - atr_ch) > 1:
        wrong.append(uid)

print(f'\nUnits with trough mismatch > 1 sample: {wrong}')
print(f'Total mismatches: {len(wrong)} / {min(30, len(wf_all))}')
