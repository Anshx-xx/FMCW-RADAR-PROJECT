"""
FMCW Radar Simulation - Phase 3A: Bistatic Geometry
=====================================================
Extends the Phase 2 monostatic baseline to a full bistatic geometry.
Baseline radar parameters are IDENTICAL to Phase 2 — only the geometry
changes (Tx ≠ Rx location), allowing a direct comparison.

  Plot 1  - Bistatic geometry diagram (Tx, Rx, Target positions)
  Plot 2  - Bistatic vs monostatic delay comparison
  Plot 3  - Beat signal: monostatic vs bistatic overlay
  Plot 4  - Range FFT: monostatic vs bistatic comparison
  Plot 5  - Baseline sweep: range response vs Tx-Rx separation
  Plot 6  - Bistatic angle effect on effective RCS
  Plot 7  - Multi-baseline range-Doppler comparison
  Plot 8  - Bistatic geometry coverage map

Run:   python phase3a_bistatic_geometry.py
Output: 8 PNG figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import sys
sys.stdout.reconfigure(encoding='utf-8')    
from scipy.signal import spectrogram

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
print("  PHASE 3A — BISTATIC GEOMETRY SIMULATION")
print(DIVIDER)
print(f"  Carrier frequency   : {fc/1e9:.0f} GHz  [LOCKED]")
print(f"  Bandwidth           : {B/1e6:.0f} MHz  [LOCKED]")
print(f"  Chirp duration      : {Tm*1e3:.1f} ms  [LOCKED]")
print(f"  Range resolution    : {range_res:.2f} m  [LOCKED]")
print(DIVIDER)


# ============================================================
# CORE DSP — inherited from Phase 2 (unchanged)
# ============================================================
def make_time_axis():
    return np.linspace(0, Tm, N_samples)

def beat_freq_from_range(R):
    return S * (2 * R / c)

def range_fft(beat_sig, n_fft=None, window=True):
    if n_fft is None:
        n_fft = N_samples * 4
    if window:
        win = np.hanning(len(beat_sig))
        beat_sig = beat_sig * win
    fft_out = np.fft.fft(beat_sig, n=n_fft)
    fft_mag = np.abs(fft_out[:n_fft // 2])
    r_axis  = np.linspace(0, max_range, n_fft // 2)
    pdb     = 20 * np.log10(fft_mag + 1e-12)
    return r_axis, fft_mag, pdb

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


# ============================================================
# BISTATIC GEOMETRY CORE FUNCTIONS
# ============================================================

def bistatic_range(tx_pos, rx_pos, target_pos):
    """
    Bistatic range equation: R_total = Rt + Rr
    tx_pos, rx_pos, target_pos: (x, y) tuples in metres.
    Returns (Rt, Rr, R_total, bistatic_angle_deg)
    """
    tx_pos     = np.array(tx_pos)
    rx_pos     = np.array(rx_pos)
    target_pos = np.array(target_pos)

    Rt = np.linalg.norm(target_pos - tx_pos)
    Rr = np.linalg.norm(target_pos - rx_pos)
    R_total = Rt + Rr

    # Bistatic angle: angle at target between Tx and Rx directions
    v1 = tx_pos - target_pos
    v2 = rx_pos - target_pos
    cos_beta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    beta_deg = np.degrees(np.arccos(np.clip(cos_beta, -1, 1)))

    return Rt, Rr, R_total, beta_deg

def generate_bistatic_beat(tx_pos, rx_pos, target_pos, snr_db=20, velocity=0.0):
    """
    Generate bistatic FMCW beat signal.
    Beat frequency determined by total bistatic path delay.
    """
    Rt, Rr, R_total, _ = bistatic_range(tx_pos, rx_pos, target_pos)
    tau_b = R_total / c          # bistatic delay (one-way Tx + one-way Rx)
    fb_b  = S * tau_b            # bistatic beat frequency
    t     = make_time_axis()
    sig   = np.cos(2 * np.pi * fb_b * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise, fb_b, tau_b, R_total

def generate_monostatic_beat(target_range, snr_db=20):
    """Phase 2 monostatic beat (for direct comparison)."""
    t   = make_time_axis()
    tau = 2 * target_range / c
    fb  = S * tau
    sig = np.cos(2 * np.pi * fb * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise, fb

def bistatic_rcs_factor(beta_deg):
    """
    Approximate bistatic RCS enhancement factor relative to monostatic.
    Strong specular enhancement near forward scatter (beta near 180°).
    Simplified model: enhancement = cos²(beta/2) scaled
    """
    beta_rad = np.radians(beta_deg)
    # Enhancement factor: peaks at beta=180° (forward scatter)
    enhancement = 1.0 + 2.0 * np.cos(beta_rad / 2) ** 4
    return enhancement


# ============================================================
# DEFAULT GEOMETRY
# ============================================================
TX_POS     = np.array([0.0,   0.0])
RX_POS     = np.array([500.0, 0.0])   # 500 m baseline
TARGET_POS = np.array([250.0, 300.0]) # target above midpoint

Rt, Rr, R_bist, beta = bistatic_range(TX_POS, RX_POS, TARGET_POS)
R_mono = np.linalg.norm(TARGET_POS - TX_POS)   # monostatic range from Tx

print(f"\n  Geometry (default):")
print(f"    Tx position       : {TX_POS}")
print(f"    Rx position       : {RX_POS}")
print(f"    Target position   : {TARGET_POS}")
print(f"    Rt (Tx→target)    : {Rt:.2f} m")
print(f"    Rr (target→Rx)    : {Rr:.2f} m")
print(f"    Bistatic R_total  : {R_bist:.2f} m")
print(f"    Monostatic R_mono : {R_mono:.2f} m")
print(f"    Bistatic angle β  : {beta:.2f}°")
print(f"    RCS enhancement   : {bistatic_rcs_factor(beta):.2f}×")


# ============================================================
# PLOT 1  —  Bistatic Geometry Diagram
# ============================================================
print("\n[1/8] Generating Plot 1: Bistatic geometry diagram...")

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle('Plot 1 — Bistatic Radar Geometry\n'
              'Tx and Rx at separate locations — bistatic path R_total = Rt + Rr',
              fontweight='bold')

# Geometry plot
ax = axes1[0]
# Draw Tx, Rx, Target
ax.scatter(*TX_POS,     s=300, marker='^', color='steelblue',  zorder=5, label='Transmitter (Tx)')
ax.scatter(*RX_POS,     s=300, marker='v', color='tomato',     zorder=5, label='Receiver (Rx)')
ax.scatter(*TARGET_POS, s=300, marker='*', color='gold',       zorder=5, label=f'Target ({TARGET_POS[0]}, {TARGET_POS[1]}) m')

# Draw signal paths
for pos, col, label in [
    (TX_POS,  'steelblue', f'Rt = {Rt:.0f} m'),
    (RX_POS,  'tomato',    f'Rr = {Rr:.0f} m'),
]:
    ax.annotate('', xy=TARGET_POS, xytext=pos,
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    mid = (np.array(pos) + TARGET_POS) / 2
    ax.text(mid[0] + 10, mid[1], label, color=col, fontsize=9, fontweight='bold')

# Baseline
ax.annotate('', xy=RX_POS, xytext=TX_POS,
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(250, -40, f'Baseline L = {np.linalg.norm(RX_POS - TX_POS):.0f} m',
        ha='center', color='gray', fontsize=9)

# Bistatic angle arc
ax.text(TARGET_POS[0] - 80, TARGET_POS[1] - 40,
        f'β = {beta:.1f}°', color='purple', fontsize=10, fontweight='bold')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'Bistatic Geometry\nR_total = {R_bist:.0f} m  |  Monostatic R = {R_mono:.0f} m')
ax.legend(loc='upper right', fontsize=8.5)
ax.set_xlim(-80, 620)
ax.set_ylim(-80, 420)
ax.set_aspect('equal')

# Comparison table
ax2 = axes1[1]
ax2.axis('off')
table_data = [
    ['Parameter',           'Monostatic',           'Bistatic'],
    ['Tx location',         '(0, 0)',                '(0, 0)'],
    ['Rx location',         '(0, 0)  [same as Tx]', '(500, 0)'],
    ['Path length',         f'2 × Rt = {2*R_mono:.0f} m',  f'Rt + Rr = {R_bist:.0f} m'],
    ['Delay τ',             f'{2*R_mono/c*1e6:.2f} µs',     f'{R_bist/c*1e6:.2f} µs'],
    ['Beat freq fb',        f'{beat_freq_from_range(R_mono)/1e3:.1f} kHz',
                             f'{S*R_bist/c/1e3:.1f} kHz'],
    ['RCS factor',          '1.0× (reference)',      f'{bistatic_rcs_factor(beta):.2f}×'],
    ['Bistatic angle β',    '0°',                    f'{beta:.1f}°'],
]
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.3, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 1:
        cell.set_facecolor('#d6eaf8')
    elif c == 2:
        cell.set_facecolor('#d5f5e3')
ax2.set_title('Monostatic vs Bistatic Parameter Comparison', pad=15)

plt.tight_layout()
plt.savefig('plot1_bistatic_geometry_diagram.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot1_bistatic_geometry_diagram.png")


# ============================================================
# PLOT 2  —  Bistatic vs Monostatic Delay Comparison
# ============================================================
print("[2/8] Generating Plot 2: Delay comparison...")

baselines   = [0, 100, 200, 500, 800, 1200]   # Tx-Rx separation (m)
target_pos_fixed = np.array([250.0, 300.0])
tx_fixed    = np.array([0.0, 0.0])

delays_bi  = []
delays_mono = []
R_totals    = []
betas       = []

for L in baselines:
    rx_pos_l = np.array([float(L), 0.0])
    _, _, R_t, b = bistatic_range(tx_fixed, rx_pos_l, target_pos_fixed)
    R_m = 2 * np.linalg.norm(target_pos_fixed - tx_fixed)   # monostatic 2-way
    delays_bi.append(R_t / c * 1e6)
    delays_mono.append(R_m / c * 1e6)
    R_totals.append(R_t)
    betas.append(b)

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Plot 2 — Bistatic vs Monostatic Delay Comparison\n'
              'As baseline L increases, bistatic delay and β both change',
              fontweight='bold')

axes2[0].plot(baselines, delays_bi,   'tomato',    lw=2, marker='o', ms=7, label='Bistatic delay (Rt+Rr)/c')
axes2[0].plot(baselines, delays_mono, 'steelblue', lw=2, ls='--', marker='s', ms=7, label='Monostatic delay 2R/c')
axes2[0].set_xlabel('Baseline L (m)')
axes2[0].set_ylabel('Time delay (µs)')
axes2[0].set_title('Signal Delay vs Baseline')
axes2[0].legend()

axes2[1].plot(baselines, R_totals, 'seagreen', lw=2, marker='D', ms=7)
axes2[1].axhline(2 * np.linalg.norm(target_pos_fixed), color='steelblue', ls='--',
                 lw=1.8, label=f'Monostatic 2R = {2*np.linalg.norm(target_pos_fixed):.0f} m')
axes2[1].set_xlabel('Baseline L (m)')
axes2[1].set_ylabel('Bistatic R_total = Rt + Rr (m)')
axes2[1].set_title('Bistatic Total Range vs Baseline')
axes2[1].legend()

axes2[2].plot(baselines, betas, 'purple', lw=2, marker='^', ms=7)
axes2[2].axhline(90, color='gray', ls=':', lw=1.5, label='90° reference')
axes2[2].set_xlabel('Baseline L (m)')
axes2[2].set_ylabel('Bistatic angle β (degrees)')
axes2[2].set_title('Bistatic Angle vs Baseline')
axes2[2].legend()

plt.tight_layout()
plt.savefig('plot2_delay_comparison.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot2_delay_comparison.png")


# ============================================================
# PLOT 3  —  Beat Signal: Monostatic vs Bistatic Overlay
# ============================================================
print("[3/8] Generating Plot 3: Beat signal overlay...")

SNR_DB3 = 20
beat_mono3, fb_mono3     = generate_monostatic_beat(R_mono, snr_db=SNR_DB3)
beat_bist3, fb_bist3, tau_b3, _ = generate_bistatic_beat(
    TX_POS, RX_POS, TARGET_POS, snr_db=SNR_DB3)

t_us3 = make_time_axis() * 1e6

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 9))
fig3.suptitle('Plot 3 — Beat Signal: Monostatic vs Bistatic Overlay\n'
              f'Same target position  |  Monostatic fb = {fb_mono3/1e3:.1f} kHz  '
              f'|  Bistatic fb = {fb_bist3/1e3:.1f} kHz',
              fontweight='bold')

n_show3 = 300
axes3[0].plot(t_us3[:n_show3], beat_mono3[:n_show3], 'steelblue', lw=1, label=f'Monostatic  fb = {fb_mono3/1e3:.1f} kHz')
axes3[0].set_title('Monostatic Beat Signal')
axes3[0].set_ylabel('Amplitude')
axes3[0].legend()

axes3[1].plot(t_us3[:n_show3], beat_bist3[:n_show3], 'tomato', lw=1, label=f'Bistatic  fb = {fb_bist3/1e3:.1f} kHz')
axes3[1].set_title('Bistatic Beat Signal')
axes3[1].set_ylabel('Amplitude')
axes3[1].legend()

axes3[2].plot(t_us3[:n_show3], beat_mono3[:n_show3], 'steelblue', lw=1, alpha=0.7, label='Monostatic')
axes3[2].plot(t_us3[:n_show3], beat_bist3[:n_show3], 'tomato',    lw=1, alpha=0.7, label='Bistatic')
axes3[2].set_title('Overlay: frequency difference is visible')
axes3[2].set_xlabel('Time (µs)')
axes3[2].set_ylabel('Amplitude')
axes3[2].legend()

plt.tight_layout()
plt.savefig('plot3_beat_signal_overlay.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot3_beat_signal_overlay.png")


# ============================================================
# PLOT 4  —  Range FFT: Monostatic vs Bistatic
# ============================================================
print("[4/8] Generating Plot 4: Range FFT comparison...")

SNR_DB4 = 18
beat_m4, _         = generate_monostatic_beat(R_mono, snr_db=SNR_DB4)
beat_b4, _, _, _   = generate_bistatic_beat(TX_POS, RX_POS, TARGET_POS, snr_db=SNR_DB4)

# For bistatic: the 'range' plotted uses bistatic R_total / 2 as equivalent range
# This makes the x-axis comparable to monostatic
r_ax_m, mag_m, pdb_m = range_fft(beat_m4)
r_ax_b, mag_b, pdb_b = range_fft(beat_b4)

# Bistatic FFT peaks at R_bist/2 equivalent range bin
R_bist_equiv = R_bist / 2   # half-path equivalent

fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.5))
fig4.suptitle('Plot 4 — Range FFT: Monostatic vs Bistatic\n'
              'Bistatic x-axis uses (Rt+Rr)/2 equivalent for fair comparison',
              fontweight='bold')

axes4[0].plot(r_ax_m, pdb_m, 'steelblue', lw=1.2, label='Monostatic')
axes4[0].plot(r_ax_b, pdb_b, 'tomato',    lw=1.2, ls='--', label='Bistatic')
axes4[0].axvline(R_mono,        color='steelblue', ls=':', lw=1.5, label=f'True mono R = {R_mono:.0f} m')
axes4[0].axvline(R_bist_equiv,  color='tomato',    ls=':', lw=1.5, label=f'Bistatic equiv R = {R_bist_equiv:.0f} m')
axes4[0].set_xlabel('Equivalent range (m)')
axes4[0].set_ylabel('Power (dB)')
axes4[0].set_title('Full Range Profile Overlay')
axes4[0].legend(fontsize=8.5)
axes4[0].set_xlim(0, max_range / 2)

# Zoomed near the peaks
zoom_w = 80
axes4[1].plot(r_ax_m, pdb_m, 'steelblue', lw=1.5, label='Monostatic')
axes4[1].plot(r_ax_b, pdb_b, 'tomato',    lw=1.5, ls='--', label='Bistatic')
axes4[1].axvline(R_mono,       color='steelblue', ls=':', lw=1.5)
axes4[1].axvline(R_bist_equiv, color='tomato',    ls=':', lw=1.5)
cx = (R_mono + R_bist_equiv) / 2
axes4[1].set_xlim(cx - zoom_w, cx + zoom_w)
axes4[1].set_xlabel('Equivalent range (m)')
axes4[1].set_ylabel('Power (dB)')
axes4[1].set_title('Zoomed: Peak positions\n(monostatic vs bistatic peaks differ due to geometry)')
axes4[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot4_range_fft_comparison.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot4_range_fft_comparison.png")


# ============================================================
# PLOT 5  —  Baseline Sweep: Range Response
# ============================================================
print("[5/8] Generating Plot 5: Baseline sweep...")

sweep_baselines = [0, 200, 500, 800, 1200]
colors5 = ['steelblue', 'tomato', 'seagreen', 'purple', 'darkorange']
target5 = np.array([350.0, 300.0])
tx5     = np.array([0.0, 0.0])
SNR5    = 18

fig5, axes5 = plt.subplots(2, 3, figsize=(17, 9))
fig5.suptitle('Plot 5 — Baseline Sweep: How Tx-Rx Separation Affects Range Response\n'
              f'Target @ {target5}  |  SNR = {SNR5} dB',
              fontweight='bold')
axes5 = axes5.flatten()

snr_peaks = []
R_totals5 = []
betas5    = []

for idx, (L, col) in enumerate(zip(sweep_baselines, colors5)):
    rx5 = np.array([float(L), 0.0])
    beat5, fb5, _, R5 = generate_bistatic_beat(tx5, rx5, target5, snr_db=SNR5)
    r5, mag5, pdb5    = range_fft(beat5)
    R5_equiv          = R5 / 2
    _, _, _, b5       = bistatic_range(tx5, rx5, target5)
    rcs_fac           = bistatic_rcs_factor(b5)
    peak_pdb          = np.max(pdb5)
    snr_peaks.append(peak_pdb)
    R_totals5.append(R5)
    betas5.append(b5)

    ax5 = axes5[idx]
    ax5.plot(r5, pdb5, color=col, lw=1.2)
    ax5.axvline(R5_equiv, color='black', ls='--', lw=1.5,
                label=f'Peak @ {R5_equiv:.0f} m equiv')
    ax5.set_xlabel('Equiv range (m)')
    ax5.set_ylabel('Power (dB)')
    L_label = 'Monostatic (L=0)' if L == 0 else f'L = {L} m'
    ax5.set_title(f'{L_label}\nβ = {b5:.1f}°  |  RCS×{rcs_fac:.2f}  |  Peak: {peak_pdb:.1f} dB')
    ax5.legend(fontsize=8)
    ax5.set_xlim(0, max_range / 2)

# Summary subplot
ax_sum = axes5[5]
ax_sum.bar(range(len(sweep_baselines)), snr_peaks, color=colors5, alpha=0.85)
ax_sum.set_xticks(range(len(sweep_baselines)))
ax_sum.set_xticklabels([f'L={L}' for L in sweep_baselines], rotation=15)
ax_sum.set_ylabel('Peak FFT power (dB)')
ax_sum.set_title('Peak SNR vs Baseline\n(Higher = better detectability)')
for i, v in enumerate(snr_peaks):
    ax_sum.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=8.5)

plt.tight_layout()
plt.savefig('plot5_baseline_sweep.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot5_baseline_sweep.png")


# ============================================================
# PLOT 6  —  Bistatic Angle Effect on RCS
# ============================================================
print("[6/8] Generating Plot 6: Bistatic angle vs RCS...")

beta_vals  = np.linspace(0, 180, 500)
rcs_factor = bistatic_rcs_factor(beta_vals)
rcs_db     = 20 * np.log10(rcs_factor)

# Compute beta for each baseline in our sweep
baselines6 = np.linspace(0, 1500, 200)
target6    = np.array([400.0, 300.0])
tx6        = np.array([0.0, 0.0])
betas6     = []
for L6 in baselines6:
    rx6 = np.array([L6, 0.0])
    if L6 == 0:
        betas6.append(0.0)
    else:
        _, _, _, b6 = bistatic_range(tx6, rx6, target6)
        betas6.append(b6)
betas6 = np.array(betas6)

fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5.5))
fig6.suptitle('Plot 6 — Bistatic Angle Effect on Radar Cross Section (RCS)\n'
              'Certain bistatic angles produce stronger returns than monostatic',
              fontweight='bold')

axes6[0].plot(beta_vals, rcs_db, 'tomato', lw=2)
axes6[0].axhline(0, color='steelblue', ls='--', lw=1.5, label='Monostatic reference (0 dB)')
axes6[0].axvline(180, color='green', ls=':', lw=1.5, label='Forward scatter (β=180°)')
axes6[0].fill_between(beta_vals, 0, rcs_db, where=(rcs_db > 0),
                      alpha=0.25, color='green', label='Enhancement over monostatic')
axes6[0].set_xlabel('Bistatic angle β (degrees)')
axes6[0].set_ylabel('RCS enhancement (dB)')
axes6[0].set_title('RCS Enhancement vs Bistatic Angle')
axes6[0].legend(fontsize=8.5)

axes6[1].plot(baselines6, betas6, 'purple', lw=2)
axes6[1].set_xlabel('Baseline L (m)')
axes6[1].set_ylabel('Bistatic angle β (degrees)')
axes6[1].set_title(f'Bistatic Angle vs Baseline\nTarget @ {target6}')

axes6[2].plot(baselines6, bistatic_rcs_factor(betas6), 'seagreen', lw=2)
axes6[2].axhline(1.0, color='gray', ls='--', lw=1.5, label='Monostatic (×1)')
axes6[2].set_xlabel('Baseline L (m)')
axes6[2].set_ylabel('RCS factor (×)')
axes6[2].set_title('RCS Factor vs Baseline\n(>1 means better than monostatic)')
axes6[2].legend()

plt.tight_layout()
plt.savefig('plot6_bistatic_angle_rcs.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot6_bistatic_angle_rcs.png")


# ============================================================
# PLOT 7  —  Multi-Baseline Range-Doppler Comparison
# ============================================================
print("[7/8] Generating Plot 7: Multi-baseline Range-Doppler maps...")

R_tgt7 = 300.0
v_tgt7 = 12.0   # m/s
SNR7   = 16

configs7 = [
    (np.array([0.,  0.]), np.array([0.,    0.]), 'Monostatic (L=0 m)'),
    (np.array([0.,  0.]), np.array([300.,  0.]), 'Bistatic L = 300 m'),
    (np.array([0.,  0.]), np.array([600.,  0.]), 'Bistatic L = 600 m'),
    (np.array([0.,  0.]), np.array([1000., 0.]), 'Bistatic L = 1000 m'),
]

target7 = np.array([R_tgt7, 200.0])

fig7, axes7 = plt.subplots(2, 2, figsize=(15, 10))
fig7.suptitle(f'Plot 7 — Multi-Baseline Range-Doppler Maps\n'
              f'Target near R = {R_tgt7} m  |  v = {v_tgt7} m/s  |  SNR = {SNR7} dB',
              fontweight='bold')
axes7 = axes7.flatten()

t7 = make_time_axis()

for idx, (tx7, rx7, cfg_name) in enumerate(configs7):
    rd_mat7 = np.zeros((N_chirps, N_samples), dtype=complex)
    for ci in range(N_chirps):
        target_now = target7 + np.array([v_tgt7 * ci * Tm, 0.0])
        Rt7, Rr7, R_t7, _ = bistatic_range(tx7, rx7, target_now)
        tau7 = R_t7 / c
        fb7  = S * tau7
        sig7 = np.cos(2 * np.pi * fb7 * t7)
        noise7 = (10 ** (-SNR7 / 20)) * np.random.randn(N_samples)
        rd_mat7[ci, :] = np.fft.fft((sig7 + noise7) * np.hanning(N_samples), N_samples)

    slow_win7 = np.hanning(N_chirps).reshape(-1, 1)
    rd_2d7    = np.fft.fftshift(np.fft.fft(rd_mat7 * slow_win7, N_chirps, axis=0), axes=0)
    rd_db7    = 20 * np.log10(np.abs(rd_2d7[:, :N_samples // 2]) + 1e-12)

    r7_ax = np.linspace(0, max_range / 2, N_samples // 2)
    v7_ax = np.linspace(-max_vel, max_vel, N_chirps)
    peak_snr7 = np.max(rd_db7)

    im7 = axes7[idx].imshow(rd_db7, aspect='auto',
                             extent=[r7_ax[0], r7_ax[-1], v7_ax[0], v7_ax[-1]],
                             origin='lower', cmap='jet',
                             vmin=np.percentile(rd_db7, 60))
    plt.colorbar(im7, ax=axes7[idx], label='dB')
    axes7[idx].set_xlabel('Range (m)')
    axes7[idx].set_ylabel('Velocity (m/s)')
    axes7[idx].set_title(f'{cfg_name}\nPeak SNR ≈ {peak_snr7:.1f} dB')

plt.tight_layout()
plt.savefig('plot7_multi_baseline_rdmap.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot7_multi_baseline_rdmap.png")


# ============================================================
# PLOT 8  —  Bistatic Coverage Map
# ============================================================
print("[8/8] Generating Plot 8: Bistatic coverage map...")

grid_x = np.linspace(-200, 1000, 150)
grid_y = np.linspace(-200, 800, 150)
XX, YY = np.meshgrid(grid_x, grid_y)

TX8 = np.array([0.,   0.])
RX8 = np.array([500., 0.])
SNR_REF_DB = 60   # reference SNR at 1 m

# Monostatic coverage: SNR ∝ 1/R^4
R_mono_map = np.sqrt((XX - TX8[0])**2 + (YY - TX8[1])**2)
snr_mono   = SNR_REF_DB - 40 * np.log10(np.maximum(R_mono_map, 1))

# Bistatic coverage: SNR ∝ 1/(Rt^2 * Rr^2)
Rt8 = np.sqrt((XX - TX8[0])**2 + (YY - TX8[1])**2)
Rr8 = np.sqrt((XX - RX8[0])**2 + (YY - RX8[1])**2)
snr_bist = SNR_REF_DB - 20 * np.log10(np.maximum(Rt8, 1)) - 20 * np.log10(np.maximum(Rr8, 1))

# RCS enhancement map
beta_map = np.zeros_like(XX)
for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        tpos = np.array([XX[i, j], YY[i, j]])
        v1 = TX8 - tpos; v2 = RX8 - tpos
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_b = np.dot(v1, v2) / (n1 * n2)
            beta_map[i, j] = np.degrees(np.arccos(np.clip(cos_b, -1, 1)))

fig8, axes8 = plt.subplots(1, 3, figsize=(18, 6))
fig8.suptitle('Plot 8 — Bistatic Coverage Maps\n'
              f'Tx @ {TX8}  |  Rx @ {RX8}  |  Detection threshold = {SNR_THRESHOLD_DB} dB',
              fontweight='bold')

cmap_snr = 'RdYlGn'
for ax, data, title in [
    (axes8[0], snr_mono,                   'Monostatic SNR Map (dB)\n[1/R^4 law — circular symmetry around Tx]'),
    (axes8[1], snr_bist,                   'Bistatic SNR Map (dB)\n[1/(Rt²Rr²) — elliptical coverage]'),
    (axes8[2], snr_bist - snr_mono,        'Bistatic Gain over Monostatic (dB)\n[Green = bistatic better]'),
]:
    vmin = -20 if 'Gain' in title else SNR_THRESHOLD_DB - 5
    vmax =  20 if 'Gain' in title else 80
    cmap = 'RdBu' if 'Gain' in title else cmap_snr
    im = ax.pcolormesh(XX, YY, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, label='dB')
    if 'Gain' not in title:
        ax.contour(XX, YY, data, levels=[SNR_THRESHOLD_DB], colors='black', linewidths=2)
    ax.scatter(*TX8, s=250, marker='^', color='white', edgecolors='black', zorder=5, label='Tx')
    ax.scatter(*RX8, s=250, marker='v', color='cyan',  edgecolors='black', zorder=5, label='Rx')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot8_coverage_map.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot8_coverage_map.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print(DIVIDER)
print("  PHASE 3A RESULTS SUMMARY — BISTATIC GEOMETRY")
print(DIVIDER)
rows = [
    ("Baseline L (default)",          "500 m"),
    ("Bistatic angle β",              f"{beta:.1f}°"),
    ("Bistatic R_total",              f"{R_bist:.1f} m"),
    ("Monostatic R (Tx→target)",      f"{R_mono:.1f} m"),
    ("Bistatic RCS factor",           f"{bistatic_rcs_factor(beta):.2f}×"),
    ("Geometry model",                "Rt + Rr bistatic path"),
    ("Beat freq (bistatic)",          f"{S*R_bist/c/1e3:.1f} kHz"),
    ("Beat freq (monostatic ref)",    f"{beat_freq_from_range(R_mono)/1e3:.1f} kHz"),
]
for name, val in rows:
    print(f"  {name:<35} {val}")
print(DIVIDER)
print()
print("  Figures saved:")
figs = [
    "plot1_bistatic_geometry_diagram.png  — Geometry diagram + table",
    "plot2_delay_comparison.png           — Bistatic vs monostatic delay",
    "plot3_beat_signal_overlay.png        — Beat signal overlay",
    "plot4_range_fft_comparison.png       — Range FFT comparison",
    "plot5_baseline_sweep.png             — Baseline sweep (6 configs)",
    "plot6_bistatic_angle_rcs.png         — Bistatic angle vs RCS",
    "plot7_multi_baseline_rdmap.png       — RD maps (4 baselines)",
    "plot8_coverage_map.png               — SNR coverage maps",
]
for i, name in enumerate(figs, 1):
    print(f"  {i}. {name}")
print()
print("  Done. Phase 3A complete — ready for Phase 3B algorithms.")
print(DIVIDER)

plt.show()