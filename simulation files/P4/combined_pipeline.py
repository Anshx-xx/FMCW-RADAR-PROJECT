"""
FMCW Radar Simulation - Phase 4: Bistatic + Enhancement Combined
=================================================================
Merges the bistatic geometry framework (Phase 3A) with the advanced
signal processing algorithms (Phase 3B) into a single unified pipeline.
All radar parameters are LOCKED from Phase 2.

This is the core experimental phase — every result is measured against
the Phase 2 monostatic baseline.

  Plot 1  - End-to-end combined pipeline block diagram (visual)
  Plot 2  - Bistatic + Enhanced FFT vs monostatic baseline
  Plot 3  - SNR improvement: geometry alone, algorithm alone, combined
  Plot 4  - Multi-baseline × algorithm combination heatmap
  Plot 5  - Enhanced Range-Doppler maps: monostatic vs bistatic+algo
  Plot 6  - Weak target detection: baseline vs combined pipeline
  Plot 7  - False alarm analysis: CA-CFAR vs OS-CFAR in bistatic env
  Plot 8  - Combined SNR gain summary (all experiments)

Run:   python phase4_combined_pipeline.py
Output: 8 PNG figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.linalg import svd

np.random.seed(42)
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f9f9f9',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# SYSTEM PARAMETERS  — LOCKED BASELINE (identical to Phase 2)
# ============================================================
c         = 3e8
fc        = 77e9
B         = 150e6
Tm        = 5.5e-3
S         = B / Tm
lambda_   = c / fc

N_samples = 1024
N_chirps  = 128
fs        = N_samples / Tm

range_res    = c / (2 * B)
max_range    = c * Tm / 2
vel_res      = lambda_ / (2 * N_chirps * Tm)
max_vel      = lambda_ / (4 * Tm)

SNR_THRESHOLD_DB = 13

DIVIDER = "=" * 60

print(DIVIDER)
print("  PHASE 4 — BISTATIC + ENHANCEMENT COMBINED PIPELINE")
print(DIVIDER)
print(f"  Carrier frequency   : {fc/1e9:.0f} GHz  [LOCKED]")
print(f"  Bandwidth           : {B/1e6:.0f} MHz  [LOCKED]")
print(f"  Chirp duration      : {Tm*1e3:.1f} ms  [LOCKED]")
print(f"  Range resolution    : {range_res:.2f} m  [LOCKED]")
print(f"  SNR threshold       : {SNR_THRESHOLD_DB} dB  [LOCKED]")
print(DIVIDER)


# ============================================================
# ALL DSP FUNCTIONS — Phase 2 baseline + Phase 3A + Phase 3B
# ============================================================
def make_time_axis():
    return np.linspace(0, Tm, N_samples)

def beat_freq_from_range(R):
    return S * (2 * R / c)

# ── Phase 2 monostatic (baseline) ────────────────────────────
def generate_monostatic_beat(target_range, snr_db=20):
    t   = make_time_axis()
    tau = 2 * target_range / c
    fb  = S * tau
    sig = np.cos(2 * np.pi * fb * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise, fb

def range_fft_baseline(beat_sig):
    """Phase 2 default: Hanning + 4× zero-padding."""
    n_fft = N_samples * 4
    win   = np.hanning(N_samples)
    fft   = np.fft.fft(beat_sig * win, n=n_fft)
    mag   = np.abs(fft[:n_fft // 2])
    r     = np.linspace(0, max_range, n_fft // 2)
    pdb   = 20 * np.log10(mag + 1e-12)
    return r, mag, pdb

def ca_cfar_1d(power_db, Tr=10, Gr=4, offset_db=8):
    N = len(power_db)
    threshold = np.full(N, np.nan)
    detected  = np.zeros(N, dtype=bool)
    for i in range(Tr + Gr, N - Tr - Gr):
        left  = power_db[i - Tr - Gr : i - Gr]
        right = power_db[i + Gr + 1  : i + Gr + Tr + 1]
        noise_est    = np.mean(np.concatenate([left, right]))
        threshold[i] = noise_est + offset_db
        detected[i]  = power_db[i] > threshold[i]
    return threshold, detected

# ── Phase 3A bistatic ────────────────────────────────────────
def bistatic_range(tx_pos, rx_pos, target_pos):
    tx_pos = np.array(tx_pos); rx_pos = np.array(rx_pos); target_pos = np.array(target_pos)
    Rt = np.linalg.norm(target_pos - tx_pos)
    Rr = np.linalg.norm(target_pos - rx_pos)
    R_total = Rt + Rr
    v1 = tx_pos - target_pos; v2 = rx_pos - target_pos
    cos_beta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    beta_deg = np.degrees(np.arccos(np.clip(cos_beta, -1, 1)))
    return Rt, Rr, R_total, beta_deg

def bistatic_rcs_factor(beta_deg):
    beta_rad = np.radians(beta_deg)
    return 1.0 + 2.0 * np.cos(beta_rad / 2) ** 4

def generate_bistatic_beat(tx_pos, rx_pos, target_pos, snr_db=20):
    Rt, Rr, R_total, beta = bistatic_range(tx_pos, rx_pos, target_pos)
    tau_b = R_total / c
    fb_b  = S * tau_b
    rcs   = bistatic_rcs_factor(beta)
    amp   = np.sqrt(rcs)
    t     = make_time_axis()
    sig   = amp * np.cos(2 * np.pi * fb_b * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise, fb_b, R_total, beta, rcs

# ── Phase 3B algorithms ──────────────────────────────────────
def range_fft_enhanced(beat_sig, pad_factor=16, window_fn=None):
    """Enhanced FFT: configurable padding + window."""
    if window_fn is None:
        window_fn = np.hanning
    n_fft = N_samples * pad_factor
    win   = window_fn(N_samples)
    fft   = np.fft.fft(beat_sig * win, n=n_fft)
    mag   = np.abs(fft[:n_fft // 2])
    r     = np.linspace(0, max_range, n_fft // 2)
    pdb   = 20 * np.log10(mag + 1e-12)
    return r, mag, pdb

def os_cfar_1d(power_db, Tr=10, Gr=4, offset_db=8, k_rank=0.75):
    N = len(power_db)
    threshold = np.full(N, np.nan)
    detected  = np.zeros(N, dtype=bool)
    total_train = 2 * Tr
    k_idx = max(0, min(total_train - 1, int(k_rank * total_train)))
    for i in range(Tr + Gr, N - Tr - Gr):
        left  = power_db[i - Tr - Gr : i - Gr]
        right = power_db[i + Gr + 1  : i + Gr + Tr + 1]
        train = np.sort(np.concatenate([left, right]))
        noise_est    = train[k_idx]
        threshold[i] = noise_est + offset_db
        detected[i]  = power_db[i] > threshold[i]
    return threshold, detected

def coherent_integration_bistatic(tx_pos, rx_pos, target_pos, n_integrate, snr_db=10):
    """Coherent integration for bistatic configuration."""
    _, _, R_total, beta, rcs = generate_bistatic_beat(tx_pos, rx_pos, target_pos, snr_db=snr_db)
    amp   = np.sqrt(rcs)
    fb    = S * R_total / c
    t     = make_time_axis()
    integrated = np.zeros(N_samples, dtype=complex)
    for _ in range(n_integrate):
        sig   = amp * np.cos(2 * np.pi * fb * t)
        noise = (10 ** (-snr_db / 20)) * np.random.randn(N_samples)
        integrated += (sig + noise)
    integrated /= n_integrate
    return range_fft_enhanced(np.real(integrated), pad_factor=16)

def peak_snr(pdb, r_axis, target_r, window_m=40):
    idx_peak = np.argmin(np.abs(r_axis - target_r))
    win = max(1, int(window_m / (r_axis[1] - r_axis[0])))
    region = pdb[max(0, idx_peak - win): idx_peak + win]
    peak   = np.max(region)
    noise_region = pdb[int(len(pdb) * 0.7):]
    noise_floor  = np.percentile(noise_region, 50)
    return peak - noise_floor


# ============================================================
# DEFAULT GEOMETRY (used throughout Phase 4)
# ============================================================
TX_POS     = np.array([0.0,   0.0])
RX_POS     = np.array([500.0, 0.0])
TARGET_POS = np.array([350.0, 280.0])

Rt, Rr, R_bist, beta = bistatic_range(TX_POS, RX_POS, TARGET_POS)
R_mono = np.linalg.norm(TARGET_POS - TX_POS)   # monostatic range from Tx
R_bist_equiv = R_bist / 2                       # equivalent monostatic range

print(f"\n  Combined pipeline geometry:")
print(f"    Monostatic range  : {R_mono:.1f} m")
print(f"    Bistatic R_total  : {R_bist:.1f} m")
print(f"    Bistatic angle β  : {beta:.1f}°")
print(f"    RCS enhancement   : {bistatic_rcs_factor(beta):.2f}×")


# ============================================================
# PLOT 1  —  Pipeline Block Diagram (visual text-art in matplotlib)
# ============================================================
print("\n[1/8] Generating Plot 1: Combined pipeline block diagram...")

fig1, ax1 = plt.subplots(figsize=(16, 7))
fig1.suptitle('Plot 1 — Phase 4: Combined Bistatic + Enhancement Pipeline\n'
              'End-to-end block diagram',
              fontweight='bold')
ax1.axis('off')

blocks = [
    (0.05, 0.65, 'SCENE\nGENERATION',     '#2c3e50', 'white',
     'Bistatic geometry\nTx/Rx positions\nTarget placement'),
    (0.22, 0.65, 'BISTATIC\nPROPAGATION', '#1a5276', 'white',
     'Rt + Rr delay\nRCS enhancement\nβ-angle model'),
    (0.39, 0.65, 'BEAT SIGNAL\nFORMATION', '#1e8449', 'white',
     'Chirp mixing\nAdditive noise\nMulti-target sum'),
    (0.56, 0.65, 'ENHANCED FFT\n(Phase 3B)',  '#7d6608', 'white',
     'Blackman window\n16× zero-padding\nSpectral shaping'),
    (0.73, 0.65, 'OS-CFAR\nDETECTION',     '#78281f', 'white',
     'Order-statistics\nAdaptive threshold\nClutter robust'),
    (0.90, 0.65, 'PERFORMANCE\nEVALUATION', '#4a235a', 'white',
     'SNR gain\nPd, FAR\nComparison'),
]

box_w = 0.14; box_h = 0.22
for (x, y, title, fc_col, tc_col, sub) in blocks:
    rect = mpatches.FancyBboxPatch((x, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0.01',
                                    facecolor=fc_col, edgecolor='white',
                                    linewidth=2, transform=ax1.transAxes)
    ax1.add_patch(rect)
    ax1.text(x + box_w / 2, y + 0.04, title,
             transform=ax1.transAxes, ha='center', va='center',
             color=tc_col, fontsize=9.5, fontweight='bold')
    ax1.text(x + box_w / 2, y - 0.07, sub,
             transform=ax1.transAxes, ha='center', va='center',
             color='#d5d8dc', fontsize=7.5)

# Arrows between blocks
for i in range(len(blocks) - 1):
    x_start = blocks[i][0]   + box_w
    x_end   = blocks[i + 1][0]
    y_arr   = blocks[i][1]
    ax1.annotate('', xy=(x_end, y_arr), xytext=(x_start, y_arr),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))

# Phase labels
for (x, y, title, *_) in blocks:
    phase_labels = ['Phase 3A', 'Phase 3A', 'Phase 2', 'Phase 3B', 'Phase 3B', 'Phase 5']
for idx, ((x, y, *_), ph) in enumerate(zip(blocks, phase_labels)):
    ax1.text(x + box_w / 2, y - box_h / 2 - 0.05, f'[{ph}]',
             transform=ax1.transAxes, ha='center', va='top',
             color='#aab7b8', fontsize=8, style='italic')

# Bottom: comparison baseline
ax1.text(0.5, 0.12,
         '▼  COMPARISON REFERENCE  ▼\n'
         'Phase 2 Monostatic Baseline: Hanning + 4× FFT + CA-CFAR\n'
         f'Parameters LOCKED: fc={fc/1e9:.0f} GHz | B={B/1e6:.0f} MHz | Tm={Tm*1e3:.1f} ms | '
         f'N={N_samples} | Nc={N_chirps}',
         transform=ax1.transAxes, ha='center', va='center',
         fontsize=10, color='#f9e79f',
         bbox=dict(boxstyle='round,pad=0.5', fc='#1c2833', ec='#f9e79f', lw=1.5))

plt.tight_layout()
plt.savefig('plot1_pipeline_diagram.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot1_pipeline_diagram.png")


# ============================================================
# PLOT 2  —  Bistatic + Enhanced FFT vs Monostatic Baseline
# ============================================================
print("[2/8] Generating Plot 2: Bistatic + enhanced vs baseline...")

SNR2 = 14
beat_mono2, _   = generate_monostatic_beat(R_mono, snr_db=SNR2)
beat_bist2, _, _, beta2, rcs2 = generate_bistatic_beat(TX_POS, RX_POS, TARGET_POS, snr_db=SNR2)

r_base2, _, pdb_base2 = range_fft_baseline(beat_mono2.copy())
r_bist_fft2, _, pdb_bist_fft2   = range_fft_baseline(beat_bist2.copy())
r_bist_enh2, _, pdb_bist_enh2   = range_fft_enhanced(beat_bist2.copy(), pad_factor=16, window_fn=np.blackman)

snr_base2     = peak_snr(pdb_base2,     r_base2,     R_mono)
snr_bist_fft2 = peak_snr(pdb_bist_fft2, r_bist_fft2, R_bist_equiv)
snr_bist_enh2 = peak_snr(pdb_bist_enh2, r_bist_enh2, R_bist_equiv)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle('Plot 2 — Bistatic + Enhanced FFT vs Phase 2 Monostatic Baseline\n'
              f'SNR = {SNR2} dB  |  Bistatic angle β = {beta:.1f}°  |  RCS factor = {rcs2:.2f}×',
              fontweight='bold')

axes2[0].plot(r_base2,     pdb_base2,     'steelblue', lw=2.0, ls='--', label=f'Mono Baseline  SNR={snr_base2:.1f} dB')
axes2[0].plot(r_bist_fft2, pdb_bist_fft2, 'tomato',    lw=1.5, label=f'Bistatic + Baseline FFT  SNR={snr_bist_fft2:.1f} dB')
axes2[0].plot(r_bist_enh2, pdb_bist_enh2, 'seagreen',  lw=1.5, label=f'Bistatic + Enhanced FFT  SNR={snr_bist_enh2:.1f} dB')
for R_m, lab, col in [(R_mono, f'Mono R={R_mono:.0f}m', 'steelblue'),
                       (R_bist_equiv, f'Bist equiv R={R_bist_equiv:.0f}m', 'tomato')]:
    axes2[0].axvline(R_m, color=col, ls=':', lw=1.5)
axes2[0].set_xlabel('Range (m)')
axes2[0].set_ylabel('Power (dB)')
axes2[0].set_title('Full Range Profiles')
axes2[0].set_xlim(0, max_range / 2)
axes2[0].legend(fontsize=8.5)

# Zoomed near target
cx2 = (R_mono + R_bist_equiv) / 2
axes2[1].plot(r_base2,     pdb_base2,     'steelblue', lw=2.0, ls='--', label='Mono Baseline')
axes2[1].plot(r_bist_fft2, pdb_bist_fft2, 'tomato',    lw=1.5, label='Bistatic + Baseline FFT')
axes2[1].plot(r_bist_enh2, pdb_bist_enh2, 'seagreen',  lw=1.5, label='Bistatic + Enhanced FFT')
axes2[1].set_xlim(cx2 - 60, cx2 + 60)
axes2[1].set_xlabel('Range (m)')
axes2[1].set_ylabel('Power (dB)')
axes2[1].set_title('Zoomed: Peak SNR comparison')
axes2[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot2_bistatic_enhanced_vs_baseline.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot2_bistatic_enhanced_vs_baseline.png")


# ============================================================
# PLOT 3  —  SNR Improvement: Geometry Alone / Algo Alone / Combined
# ============================================================
print("[3/8] Generating Plot 3: SNR gain decomposition...")

SNR3    = 12
N_RUNS3 = 20

def run_snr(fn, n_runs=N_RUNS3):
    """Average peak SNR over n_runs Monte Carlo trials."""
    return np.mean([fn() for _ in range(n_runs)])

conditions3 = [
    ('Phase 2 Baseline\n(Mono + Baseline FFT)',
     lambda: peak_snr(*range_fft_baseline(generate_monostatic_beat(R_mono, snr_db=SNR3)[0]), R_mono)),
    ('Bistatic only\n(Bistatic + Baseline FFT)',
     lambda: peak_snr(*range_fft_baseline(generate_bistatic_beat(TX_POS, RX_POS, TARGET_POS, snr_db=SNR3)[0]),
                      R_bist_equiv)),
    ('Algorithm only\n(Mono + Enhanced FFT)',
     lambda: peak_snr(*range_fft_enhanced(generate_monostatic_beat(R_mono, snr_db=SNR3)[0], pad_factor=16, window_fn=np.blackman),
                      R_mono)),
    ('Combined\n(Bistatic + Enhanced FFT)',
     lambda: peak_snr(*range_fft_enhanced(generate_bistatic_beat(TX_POS, RX_POS, TARGET_POS, snr_db=SNR3)[0],
                                          pad_factor=16, window_fn=np.blackman),
                      R_bist_equiv)),
    ('Full Combined\n(Bistatic + Enh + Coh N=32)',
     lambda: peak_snr(*coherent_integration_bistatic(TX_POS, RX_POS, TARGET_POS, 32, snr_db=SNR3),
                      R_bist_equiv)),
]

snr3_vals = [run_snr(fn) for _, fn in conditions3]
labels3   = [c[0] for c in conditions3]
baseline3 = snr3_vals[0]
gains3    = [v - baseline3 for v in snr3_vals]
colors3   = ['steelblue', 'tomato', 'darkorange', 'seagreen', 'purple']

fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle(f'Plot 3 — SNR Gain Decomposition: Geometry vs Algorithm vs Combined\n'
              f'SNR input = {SNR3} dB  |  {N_RUNS3} Monte Carlo runs',
              fontweight='bold')

bars3a = axes3[0].bar(range(len(conditions3)), snr3_vals, color=colors3, alpha=0.85, width=0.6)
axes3[0].axhline(baseline3, color='steelblue', ls='--', lw=2, label=f'Baseline: {baseline3:.1f} dB')
axes3[0].axhline(SNR_THRESHOLD_DB, color='red', ls=':', lw=1.5, label=f'Detection threshold: {SNR_THRESHOLD_DB} dB')
axes3[0].set_xticks(range(len(conditions3)))
axes3[0].set_xticklabels(labels3, rotation=10, ha='right', fontsize=8.5)
axes3[0].set_ylabel('Peak SNR (dB)')
axes3[0].set_title('Absolute SNR per Configuration')
for bar, v in zip(bars3a, snr3_vals):
    axes3[0].text(bar.get_x() + bar.get_width() / 2, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)
axes3[0].legend(fontsize=8.5)

bars3b = axes3[1].bar(range(len(conditions3)), gains3, color=colors3, alpha=0.85, width=0.6)
axes3[1].axhline(0, color='black', ls='--', lw=1.5, label='Baseline (0 gain)')
axes3[1].set_xticks(range(len(conditions3)))
axes3[1].set_xticklabels(labels3, rotation=10, ha='right', fontsize=8.5)
axes3[1].set_ylabel('SNR Gain vs Phase 2 Baseline (dB)')
axes3[1].set_title('SNR Gain per Configuration\n(Proves geometry + algo are complementary)')
for bar, v in zip(bars3b, gains3):
    col = 'green' if v >= 0 else 'red'
    axes3[1].text(bar.get_x() + bar.get_width() / 2, v + 0.1 if v >= 0 else v - 0.4,
                  f'{v:+.1f} dB', ha='center', fontsize=9, fontweight='bold', color=col)
axes3[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot3_snr_gain_decomposition.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot3_snr_gain_decomposition.png")


# ============================================================
# PLOT 4  —  Heatmap: Baseline × Algorithm Combinations
# ============================================================
print("[4/8] Generating Plot 4: Combination heatmap...")

baselines4 = [0, 200, 400, 600, 800, 1000]
algorithms4 = ['Baseline FFT\n(4× pad)', 'Enhanced FFT\n(16× pad)', 'Enh FFT +\nCoh N=16']
target4     = np.array([350.0, 280.0])
tx4         = np.array([0., 0.])
SNR4        = 12
N_RUNS4     = 15

snr_heatmap = np.zeros((len(algorithms4), len(baselines4)))
gain_heatmap = np.zeros_like(snr_heatmap)

# Reference: monostatic + baseline FFT
ref_snr4 = np.mean([
    peak_snr(*range_fft_baseline(generate_monostatic_beat(R_mono, snr_db=SNR4)[0]), R_mono)
    for _ in range(N_RUNS4)
])

for b_idx, L4 in enumerate(baselines4):
    rx4 = np.array([float(L4), 0.])
    for a_idx, algo in enumerate(algorithms4):
        snr_vals = []
        for _ in range(N_RUNS4):
            beat4, _, R4_total, b4, _ = generate_bistatic_beat(tx4, rx4, target4, snr_db=SNR4)
            R4_equiv = R4_total / 2
            if algo.startswith('Baseline'):
                r4, _, pdb4 = range_fft_baseline(beat4)
            elif algo.startswith('Enhanced'):
                r4, _, pdb4 = range_fft_enhanced(beat4, pad_factor=16, window_fn=np.blackman)
            else:  # Coherent
                r4, _, pdb4 = coherent_integration_bistatic(tx4, rx4, target4, 16, snr_db=SNR4)
            snr_vals.append(peak_snr(pdb4, r4, R4_equiv))
        snr_heatmap[a_idx, b_idx]  = np.mean(snr_vals)
        gain_heatmap[a_idx, b_idx] = np.mean(snr_vals) - ref_snr4

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 5))
fig4.suptitle(f'Plot 4 — Heatmap: Baseline Length × Algorithm Combination\n'
              f'SNR input = {SNR4} dB  |  Reference (Phase 2 Mono baseline) = {ref_snr4:.1f} dB',
              fontweight='bold')

for ax, data, title, cmap, fmt in [
    (axes4[0], snr_heatmap,  'Absolute Peak SNR (dB)', 'YlOrRd', '.1f'),
    (axes4[1], gain_heatmap, 'SNR Gain vs Phase 2 Baseline (dB)', 'RdYlGn', '+.1f'),
]:
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='dB')
    ax.set_xticks(range(len(baselines4)))
    ax.set_xticklabels([f'{L} m' for L in baselines4])
    ax.set_yticks(range(len(algorithms4)))
    ax.set_yticklabels(algorithms4, fontsize=8.5)
    ax.set_xlabel('Bistatic Baseline L')
    ax.set_ylabel('Processing Algorithm')
    ax.set_title(title)
    for i in range(len(algorithms4)):
        for j in range(len(baselines4)):
            ax.text(j, i, f'{data[i,j]:{fmt}}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if abs(data[i,j]) > (0.5 * np.max(np.abs(data))) else 'black')

plt.tight_layout()
plt.savefig('plot4_combination_heatmap.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot4_combination_heatmap.png")


# ============================================================
# PLOT 5  —  Enhanced Range-Doppler Maps
# ============================================================
print("[5/8] Generating Plot 5: Enhanced Range-Doppler maps...")

R_tgt5 = 280.0; v_tgt5 = 10.0
SNR5    = 14

configs5 = [
    ('Phase 2 Mono Baseline',         np.array([0., 0.]), np.array([0., 0.])),
    ('Bistatic L=500m + Enhanced FFT', np.array([0., 0.]), np.array([500., 0.])),
]

target5 = np.array([R_tgt5, 200.0])
fig5, axes5 = plt.subplots(1, 2, figsize=(15, 6))
fig5.suptitle(f'Plot 5 — Range-Doppler Maps: Baseline vs Combined Enhancement\n'
              f'Target R ≈ {R_tgt5} m  |  v = {v_tgt5} m/s  |  SNR = {SNR5} dB',
              fontweight='bold')

for ax, (cfg_name, tx5, rx5) in zip(axes5, configs5):
    is_mono = np.allclose(tx5, rx5)
    rd_mat5 = np.zeros((N_chirps, N_samples), dtype=complex)
    t5 = make_time_axis()
    for ci in range(N_chirps):
        target_now = target5 + np.array([v_tgt5 * ci * Tm, 0.])
        if is_mono:
            R_now = np.linalg.norm(target_now - tx5)
            fb5   = beat_freq_from_range(R_now)
            amp5  = 1.0
        else:
            Rt5, Rr5, Rt5_total, b5 = bistatic_range(tx5, rx5, target_now)
            fb5  = S * Rt5_total / c
            amp5 = np.sqrt(bistatic_rcs_factor(b5))
        sig5  = amp5 * np.cos(2 * np.pi * fb5 * t5)
        noise5 = (10 ** (-SNR5 / 20)) * np.random.randn(N_samples)
        # Enhanced window for bistatic, baseline for mono
        win5 = np.blackman(N_samples) if not is_mono else np.hanning(N_samples)
        rd_mat5[ci, :] = np.fft.fft((sig5 + noise5) * win5, N_samples)

    slow_win5 = np.hanning(N_chirps).reshape(-1, 1)
    rd_2d5    = np.fft.fftshift(np.fft.fft(rd_mat5 * slow_win5, N_chirps, axis=0), axes=0)
    rd_db5    = 20 * np.log10(np.abs(rd_2d5[:, :N_samples // 2]) + 1e-12)
    r5_ax     = np.linspace(0, max_range / 2, N_samples // 2)
    v5_ax     = np.linspace(-max_vel, max_vel, N_chirps)
    peak5     = np.max(rd_db5)

    im5 = ax.imshow(rd_db5, aspect='auto',
                    extent=[r5_ax[0], r5_ax[-1], v5_ax[0], v5_ax[-1]],
                    origin='lower', cmap='jet',
                    vmin=np.percentile(rd_db5, 60))
    plt.colorbar(im5, ax=ax, label='dB')
    ax.plot(R_tgt5, v_tgt5, 'w+', ms=14, mew=2.5, label=f'Target ({R_tgt5} m, {v_tgt5} m/s)')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'{cfg_name}\nPeak ≈ {peak5:.1f} dB')
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot5_enhanced_rdmap.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot5_enhanced_rdmap.png")


# ============================================================
# PLOT 6  —  Weak Target Detection
# ============================================================
print("[6/8] Generating Plot 6: Weak target detection...")

R6_weak   = 500.0     # long range — very weak
SNR6_vals = [3, 6, 9, 12, 15, 18]
N_TRIALS6 = 30

R6_equiv = bistatic_range(TX_POS, RX_POS,
                           np.array([R6_weak / np.sqrt(2), R6_weak / np.sqrt(2)]))[2] / 2

pipelines6 = {
    'Phase 2 Baseline\n(Mono + CA-CFAR)': [],
    'Combined Pipeline\n(Bistatic + Enh + OS-CFAR)': [],
}
win6_window = 20

for snr6 in SNR6_vals:
    # Baseline
    pd_base6 = 0
    for _ in range(N_TRIALS6):
        beat6m, _ = generate_monostatic_beat(R6_weak, snr_db=snr6)
        r6m, _, pdb6m = range_fft_baseline(beat6m)
        thr6m, det6m = ca_cfar_1d(pdb6m, Tr=12, Gr=5, offset_db=8)
        idx6m = np.argmin(np.abs(r6m - R6_weak))
        if np.any(det6m[max(0, idx6m - win6_window): idx6m + win6_window]):
            pd_base6 += 1
    pipelines6['Phase 2 Baseline\n(Mono + CA-CFAR)'].append(pd_base6 / N_TRIALS6)

    # Combined
    pd_comb6 = 0
    t_pos6 = np.array([R6_weak * 0.6, R6_weak * 0.4])
    for _ in range(N_TRIALS6):
        beat6b, _, R6b_total, _, _ = generate_bistatic_beat(TX_POS, RX_POS, t_pos6, snr_db=snr6)
        r6b, _, pdb6b = range_fft_enhanced(beat6b, pad_factor=16, window_fn=np.blackman)
        thr6b, det6b = os_cfar_1d(pdb6b, Tr=12, Gr=5, offset_db=8, k_rank=0.75)
        idx6b = np.argmin(np.abs(r6b - R6b_total / 2))
        if np.any(det6b[max(0, idx6b - win6_window): idx6b + win6_window]):
            pd_comb6 += 1
    pipelines6['Combined Pipeline\n(Bistatic + Enh + OS-CFAR)'].append(pd_comb6 / N_TRIALS6)

fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5.5))
fig6.suptitle(f'Plot 6 — Weak Target Detection: Baseline vs Combined Pipeline\n'
              f'Target at long range  |  {N_TRIALS6} Monte Carlo trials per point',
              fontweight='bold')

colors6 = ['steelblue', 'seagreen']
for (pipe_name, pd_vals), col in zip(pipelines6.items(), colors6):
    ls = '--' if 'Baseline' in pipe_name else '-'
    axes6[0].plot(SNR6_vals, pd_vals, lw=2.5, ls=ls, color=col,
                  marker='o', ms=7, label=pipe_name.replace('\n', ' '))
axes6[0].axhline(0.9, color='black', ls=':', lw=1.5, label='Pd = 0.9 target')
axes6[0].set_xlabel('Input SNR (dB)')
axes6[0].set_ylabel('Probability of Detection (Pd)')
axes6[0].set_title('Pd vs SNR: Baseline vs Combined')
axes6[0].set_ylim(0, 1.05)
axes6[0].legend(fontsize=8.5)

# Find SNR required for Pd=0.9
for (pipe_name, pd_vals), col in zip(pipelines6.items(), colors6):
    pd_arr = np.array(pd_vals)
    above  = np.where(pd_arr >= 0.9)[0]
    snr_req = SNR6_vals[above[0]] if len(above) > 0 else '>18'
    axes6[1].bar(pipe_name.replace('\n', ' '), snr_req if isinstance(snr_req, (int, float)) else 20,
                 color=col, alpha=0.85, width=0.5)
    axes6[1].text(pipe_name.replace('\n', ' '),
                  (snr_req if isinstance(snr_req, (int, float)) else 20) + 0.2,
                  f'SNR ≥ {snr_req} dB', ha='center', fontsize=9, fontweight='bold')
axes6[1].set_ylabel('Min SNR for Pd = 0.9 (dB)\n(Lower = better)')
axes6[1].set_title('Minimum SNR Required for 90% Detection\n(Combined pipeline needs less SNR)')
axes6[1].tick_params(axis='x', labelsize=8.5)

plt.tight_layout()
plt.savefig('plot6_weak_target_detection.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot6_weak_target_detection.png")


# ============================================================
# PLOT 7  —  False Alarm Analysis: CA-CFAR vs OS-CFAR
# ============================================================
print("[7/8] Generating Plot 7: False alarm analysis...")

TARGET_R7  = 300.0
CLUTTER_R7 = 260.0
SNR7_tgt   = 12
SNR7_clut  = 28
N_TRIALS7  = 25
offset_vals7 = np.arange(4, 18, 1)

pd_ca7 = []; pfa_ca7 = []
pd_os7 = []; pfa_os7 = []

for off7 in offset_vals7:
    pd_ca_sum = pfa_ca_sum = 0
    pd_os_sum = pfa_os_sum = 0
    for _ in range(N_TRIALS7):
        # Bistatic beat with clutter + target
        tx7 = np.array([0., 0.]); rx7 = np.array([500., 0.])
        tgt7 = np.array([TARGET_R7, 200.])
        clt7 = np.array([CLUTTER_R7, 200.])
        beat_tgt7, _, Rt7, _, _ = generate_bistatic_beat(tx7, rx7, tgt7, snr_db=SNR7_tgt)
        beat_clt7, _, Rc7, _, _ = generate_bistatic_beat(tx7, rx7, clt7, snr_db=SNR7_clut)
        combined7 = beat_tgt7 + beat_clt7
        r7, _, pdb7 = range_fft_enhanced(combined7, pad_factor=16)
        thr_ca7, det_ca7 = ca_cfar_1d(pdb7, Tr=12, Gr=5, offset_db=off7)
        thr_os7, det_os7 = os_cfar_1d(pdb7, Tr=12, Gr=5, offset_db=off7, k_rank=0.75)

        t_idx7 = np.argmin(np.abs(r7 - Rt7 / 2))
        win7   = 20
        pd_ca_sum  += int(np.any(det_ca7[max(0, t_idx7 - win7): t_idx7 + win7]))
        pd_os_sum  += int(np.any(det_os7[max(0, t_idx7 - win7): t_idx7 + win7]))
        fa_mask7 = np.ones(len(det_ca7), dtype=bool)
        fa_mask7[max(0, t_idx7 - 30): t_idx7 + 30] = False
        pfa_ca_sum += det_ca7[fa_mask7].sum() / fa_mask7.sum()
        pfa_os_sum += det_os7[fa_mask7].sum() / fa_mask7.sum()

    pd_ca7.append(pd_ca_sum / N_TRIALS7)
    pd_os7.append(pd_os_sum / N_TRIALS7)
    pfa_ca7.append(pfa_ca_sum / N_TRIALS7)
    pfa_os7.append(pfa_os_sum / N_TRIALS7)

fig7, axes7 = plt.subplots(1, 3, figsize=(17, 5.5))
fig7.suptitle(f'Plot 7 — False Alarm Analysis in Bistatic Clutter Environment\n'
              f'Target @ ~{TARGET_R7} m  |  Clutter @ ~{CLUTTER_R7} m (SNR={SNR7_clut} dB)',
              fontweight='bold')

axes7[0].plot(offset_vals7, pd_ca7, 'steelblue', lw=2, marker='o', ms=6, ls='--', label='CA-CFAR (P2 baseline)')
axes7[0].plot(offset_vals7, pd_os7, 'tomato',    lw=2, marker='s', ms=6, label='OS-CFAR (enhanced)')
axes7[0].axhline(0.9, color='black', ls=':', lw=1.5)
axes7[0].set_xlabel('CFAR offset (dB)')
axes7[0].set_ylabel('Pd')
axes7[0].set_title('Detection Probability vs CFAR offset')
axes7[0].set_ylim(0, 1.05)
axes7[0].legend()

axes7[1].plot(offset_vals7, pfa_ca7, 'steelblue', lw=2, marker='o', ms=6, ls='--', label='CA-CFAR')
axes7[1].plot(offset_vals7, pfa_os7, 'tomato',    lw=2, marker='s', ms=6, label='OS-CFAR')
axes7[1].set_xlabel('CFAR offset (dB)')
axes7[1].set_ylabel('False alarm rate')
axes7[1].set_title('False Alarm Rate vs CFAR offset')
axes7[1].legend()

axes7[2].plot(pfa_ca7, pd_ca7, 'steelblue', lw=2, marker='o', ms=6, ls='--', label='CA-CFAR (P2 baseline)')
axes7[2].plot(pfa_os7, pd_os7, 'tomato',    lw=2, marker='s', ms=6, label='OS-CFAR (enhanced)')
axes7[2].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random guess')
axes7[2].set_xlabel('False alarm rate')
axes7[2].set_ylabel('Pd')
axes7[2].set_title('ROC Curve: CA-CFAR vs OS-CFAR\n(Bistatic environment)')
axes7[2].set_xlim(0, 1); axes7[2].set_ylim(0, 1.05)
axes7[2].legend()

plt.tight_layout()
plt.savefig('plot7_false_alarm_analysis.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot7_false_alarm_analysis.png")


# ============================================================
# PLOT 8  —  Combined SNR Gain Summary
# ============================================================
print("[8/8] Generating Plot 8: Combined SNR gain summary...")

SNR8    = 10   # very challenging — makes gains visible
N_RUNS8 = 20

target8 = np.array([350., 280.])
R8_mono = np.linalg.norm(target8)
_, _, R8_bist, _ = bistatic_range(TX_POS, RX_POS, target8)
R8_bist_eq = R8_bist / 2

experiments8 = [
    ('P2 Baseline\nMono+Baseline FFT',
     lambda: peak_snr(*range_fft_baseline(generate_monostatic_beat(R8_mono, snr_db=SNR8)[0]), R8_mono)),
    ('Geo only\nBistatic+Baseline FFT',
     lambda: peak_snr(*range_fft_baseline(generate_bistatic_beat(TX_POS, RX_POS, target8, snr_db=SNR8)[0]),
                      R8_bist_eq)),
    ('Algo only\nMono+Enh FFT(16×)',
     lambda: peak_snr(*range_fft_enhanced(generate_monostatic_beat(R8_mono, snr_db=SNR8)[0],
                                          pad_factor=16, window_fn=np.blackman), R8_mono)),
    ('Geo+Algo\nBistatic+Enh FFT',
     lambda: peak_snr(*range_fft_enhanced(generate_bistatic_beat(TX_POS, RX_POS, target8, snr_db=SNR8)[0],
                                          pad_factor=16, window_fn=np.blackman), R8_bist_eq)),
    ('Geo+Algo+Coh N=16\nBistatic+Enh+Integ',
     lambda: peak_snr(*coherent_integration_bistatic(TX_POS, RX_POS, target8, 16, snr_db=SNR8), R8_bist_eq)),
    ('Geo+Algo+Coh N=64\nBistatic+Enh+Integ',
     lambda: peak_snr(*coherent_integration_bistatic(TX_POS, RX_POS, target8, 64, snr_db=SNR8), R8_bist_eq)),
]

snr8_vals = [np.mean([fn() for _ in range(N_RUNS8)]) for _, fn in experiments8]
labels8   = [e[0] for e in experiments8]
baseline8 = snr8_vals[0]
gains8    = [v - baseline8 for v in snr8_vals]
colors8   = ['steelblue', 'tomato', 'darkorange', 'seagreen', 'purple', 'darkblue']

fig8, axes8 = plt.subplots(1, 2, figsize=(16, 6.5))
fig8.suptitle(f'Plot 8 — Combined SNR Gain Summary: All Phase 4 Experiments\n'
              f'Input SNR = {SNR8} dB  |  {N_RUNS8} Monte Carlo runs  |  '
              f'Phase 2 Baseline = {baseline8:.1f} dB',
              fontweight='bold')

bars8a = axes8[0].bar(range(len(experiments8)), snr8_vals, color=colors8, alpha=0.85, width=0.65)
axes8[0].axhline(baseline8, color='steelblue', ls='--', lw=2, label=f'Phase 2 Baseline: {baseline8:.1f} dB')
axes8[0].axhline(SNR_THRESHOLD_DB, color='red', ls=':', lw=2, label=f'Detection threshold: {SNR_THRESHOLD_DB} dB')
axes8[0].set_xticks(range(len(experiments8)))
axes8[0].set_xticklabels([l.split('\n')[0] for l in labels8], rotation=15, ha='right', fontsize=8)
axes8[0].set_ylabel('Peak SNR (dB)')
axes8[0].set_title('Absolute Peak SNR — All Phase 4 Configurations')
for bar, v in zip(bars8a, snr8_vals):
    axes8[0].text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                  f'{v:.1f}', ha='center', fontsize=8.5, fontweight='bold')
axes8[0].legend(fontsize=8.5)

bars8b = axes8[1].bar(range(len(experiments8)), gains8, color=colors8, alpha=0.85, width=0.65)
axes8[1].axhline(0, color='black', ls='--', lw=2, label='Phase 2 Baseline (0 dB)')
axes8[1].set_xticks(range(len(experiments8)))
axes8[1].set_xticklabels([l.split('\n')[0] for l in labels8], rotation=15, ha='right', fontsize=8)
axes8[1].set_ylabel('SNR Gain vs Phase 2 Baseline (dB)')
axes8[1].set_title('SNR Gain per Pipeline Configuration\n(Phase 4 proof: combined > individual)')
for bar, v in zip(bars8b, gains8):
    col = 'green' if v >= 0 else 'red'
    axes8[1].text(bar.get_x() + bar.get_width() / 2,
                  v + 0.1 if v >= 0 else v - 0.5,
                  f'{v:+.1f} dB', ha='center', fontsize=9, fontweight='bold', color=col)
axes8[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot8_combined_snr_summary.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot8_combined_snr_summary.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print(DIVIDER)
print("  PHASE 4 RESULTS SUMMARY — COMBINED PIPELINE")
print(DIVIDER)
print(f"  {'Configuration':<40} {'Peak SNR':>9}  {'Gain':>9}")
print("-" * 62)
for lbl, snr_v, gain_v in zip(labels8, snr8_vals, gains8):
    lbl_flat = lbl.replace('\n', ' ')
    print(f"  {lbl_flat:<40} {snr_v:>7.1f} dB  {gain_v:>+7.1f} dB")
print(DIVIDER)
print(f"\n  Phase 2 Monostatic Reference SNR : {baseline8:.1f} dB")
print(f"  Max combined SNR                 : {max(snr8_vals):.1f} dB")
print(f"  Maximum gain over baseline       : {max(gains8):+.1f} dB")
print(DIVIDER)
print()
print("  Figures saved:")
figs = [
    "plot1_pipeline_diagram.png           — End-to-end block diagram",
    "plot2_bistatic_enhanced_vs_baseline.png— Range FFT comparison",
    "plot3_snr_gain_decomposition.png     — Geo/Algo/Combined gain",
    "plot4_combination_heatmap.png        — Baseline × algorithm matrix",
    "plot5_enhanced_rdmap.png             — Range-Doppler comparison",
    "plot6_weak_target_detection.png      — Weak target Pd vs SNR",
    "plot7_false_alarm_analysis.png       — ROC: CA vs OS-CFAR",
    "plot8_combined_snr_summary.png       — All experiments summary",
]
for i, name in enumerate(figs, 1):
    print(f"  {i}. {name}")
print()
print("  Done. Phase 4 complete — ready for Phase 5 evaluation.")
print(DIVIDER)

plt.show()