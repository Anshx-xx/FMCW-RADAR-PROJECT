"""
FMCW Radar Simulation - Phase 3B: Algorithms for Range Enhancement
===================================================================
Implements advanced signal processing algorithms on top of the
Phase 2 monostatic baseline. All radar parameters are LOCKED.
Each algorithm is benchmarked directly against the Phase 2 baseline FFT.

  Plot 1  - Zero-padding FFT: resolution vs padding factor
  Plot 2  - Windowing + zero-padding combined: peak sharpness comparison
  Plot 3  - MUSIC super-resolution: separating closely spaced targets
  Plot 4  - OS-CFAR vs CA-CFAR: performance in clutter
  Plot 5  - Coherent signal integration: SNR improvement vs chirps
  Plot 6  - Sparse / OMP reconstruction: high-resolution from few samples
  Plot 7  - Algorithm benchmark: detection probability & peak sharpness
  Plot 8  - SNR gain summary: all enhancement methods vs Phase 2 baseline

Run:   python phase3b_range_enhancement_algorithms.py
Output: 8 PNG figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.stdout.reconfigure(encoding='utf-8')
from scipy.signal import spectrogram
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
print("  PHASE 3B — RANGE ENHANCEMENT ALGORITHMS")
print(DIVIDER)
print(f"  Carrier frequency   : {fc/1e9:.0f} GHz  [LOCKED]")
print(f"  Bandwidth           : {B/1e6:.0f} MHz  [LOCKED]")
print(f"  Chirp duration      : {Tm*1e3:.1f} ms  [LOCKED]")
print(f"  Range resolution    : {range_res:.2f} m  [LOCKED]")
print(f"  SNR threshold       : {SNR_THRESHOLD_DB} dB  [LOCKED]")
print(DIVIDER)


# ============================================================
# CORE DSP — inherited from Phase 2 (unchanged)
# ============================================================
def make_time_axis():
    return np.linspace(0, Tm, N_samples)

def beat_freq_from_range(R):
    return S * (2 * R / c)

def generate_beat_signal(target_range, snr_db=20):
    t   = make_time_axis()
    tau = 2 * target_range / c
    fb  = S * tau
    sig = np.cos(2 * np.pi * fb * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise

def generate_multi_target_beat(targets_ranges, targets_snr):
    t   = make_time_axis()
    sig = np.zeros(N_samples)
    for R, snr in zip(targets_ranges, targets_snr):
        tau = 2 * R / c
        fb  = S * tau
        amp = 10 ** (snr / 20)
        sig += amp * np.cos(2 * np.pi * fb * t)
    noise = np.random.randn(N_samples)
    return sig + noise

def range_fft(beat_sig, n_fft=None, window=True):
    """Phase 2 baseline FFT — unchanged."""
    if n_fft is None:
        n_fft = N_samples * 4      # Phase 2 default: 4× zero-padding
    if window:
        win = np.hanning(len(beat_sig))
        beat_sig = beat_sig * win
    fft_out = np.fft.fft(beat_sig, n=n_fft)
    fft_mag = np.abs(fft_out[:n_fft // 2])
    r_axis  = np.linspace(0, max_range, n_fft // 2)
    pdb     = 20 * np.log10(fft_mag + 1e-12)
    return r_axis, fft_mag, pdb

def ca_cfar_1d(power_db, Tr=10, Gr=4, offset_db=8):
    """Phase 2 CA-CFAR — unchanged."""
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
# ENHANCEMENT ALGORITHM FUNCTIONS
# ============================================================

def range_fft_zeropad(beat_sig, pad_factor=1, window=True):
    """
    Zero-padded FFT.
    pad_factor: multiplier on N_samples (1=no padding, 4=Phase2 default, 16=enhanced).
    """
    n_fft = N_samples * pad_factor
    if window:
        win = np.hanning(len(beat_sig))
        beat_sig = beat_sig * win
    fft_out = np.fft.fft(beat_sig, n=n_fft)
    fft_mag = np.abs(fft_out[:n_fft // 2])
    r_axis  = np.linspace(0, max_range, n_fft // 2)
    pdb     = 20 * np.log10(fft_mag + 1e-12)
    return r_axis, fft_mag, pdb

def os_cfar_1d(power_db, Tr=10, Gr=4, offset_db=8, k_rank=0.75):
    """
    Order-Statistics CFAR (OS-CFAR).
    Uses the k-th order statistic of training cells as noise estimate.
    More robust than CA-CFAR in multi-target / clutter environments.
    k_rank: fraction (0-1) of sorted training cells to use as noise estimate.
    """
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

def music_pseudospectrum(beat_sig, n_sources=2, n_range_bins=2048):
    """
    MUSIC (MUltiple SIgnal Classification) algorithm adapted for range estimation.
    Uses the covariance matrix of overlapping windows to separate closely
    spaced targets that conventional FFT cannot resolve.
    Returns: (range_axis, MUSIC_pseudospectrum_dB)
    """
    t    = make_time_axis()
    L    = 64    # snapshot length
    K    = N_samples - L + 1
    # Build data matrix: K snapshots × L samples
    X = np.array([beat_sig[i:i+L] for i in range(K)])
    # Covariance matrix
    Rxx = (X.T @ np.conj(X)) / K
    # SVD / eigendecomposition
    U, sigma, Vh = svd(Rxx)
    # Noise subspace: columns L-n_sources onwards
    En = U[:, n_sources:]   # noise eigenvectors
    # Steering vectors for each candidate range
    r_cands = np.linspace(0, max_range / 2, n_range_bins)
    fb_cands = beat_freq_from_range(r_cands)
    t_L = t[:L]
    music_spec = np.zeros(n_range_bins)
    for j, fb_c in enumerate(fb_cands):
        a = np.exp(1j * 2 * np.pi * fb_c * t_L)
        denom = np.real(np.conj(a) @ En @ np.conj(En).T @ a)
        music_spec[j] = 1.0 / (denom + 1e-30)
    music_db = 20 * np.log10(music_spec / np.max(music_spec) + 1e-12)
    return r_cands, music_db

def coherent_integration(target_range, n_integrate, snr_db=10):
    """
    Coherently integrate n_integrate chirps to improve SNR.
    SNR improvement ≈ 10*log10(n_integrate) dB.
    Returns: range_fft spectrum (dB) after integration.
    """
    integrated = np.zeros(N_samples, dtype=complex)
    t  = make_time_axis()
    fb = beat_freq_from_range(target_range)
    for _ in range(n_integrate):
        sig   = np.cos(2 * np.pi * fb * t)
        noise = (10 ** (-snr_db / 20)) * np.random.randn(N_samples)
        # Coherent: signals add in phase, noise averages out
        integrated += (sig + noise)
    # Average and FFT
    integrated /= n_integrate
    win = np.hanning(N_samples)
    n_fft = N_samples * 4
    fft_out = np.abs(np.fft.fft(integrated * win, n=n_fft)[:n_fft // 2])
    r_axis  = np.linspace(0, max_range, n_fft // 2)
    return r_axis, 20 * np.log10(fft_out + 1e-12)

def omp_sparse_reconstruction(beat_sig, n_range_atoms=512, sparsity=3):
    """
    Orthogonal Matching Pursuit (OMP) — sparse signal reconstruction.
    Represents the beat signal as a sparse sum of range-frequency atoms.
    Returns: (range_axis_sparse, sparse_spectrum)
    """
    t = make_time_axis()
    r_atoms = np.linspace(1, max_range / 2, n_range_atoms)
    fb_atoms = beat_freq_from_range(r_atoms)
    # Build dictionary: each column is a steering vector
    D = np.array([np.cos(2 * np.pi * fb * t) for fb in fb_atoms]).T  # N × M
    D = D / (np.linalg.norm(D, axis=0) + 1e-12)   # normalise columns

    y = beat_sig.copy()
    residual = y.copy()
    support  = []
    coeffs   = np.zeros(n_range_atoms)

    for _ in range(sparsity):
        correlations = np.abs(D.T @ residual)
        best_idx     = np.argmax(correlations)
        if best_idx not in support:
            support.append(best_idx)
        # Least squares on support
        D_s    = D[:, support]
        x_s, _, _, _ = np.linalg.lstsq(D_s, y, rcond=None)
        residual = y - D_s @ x_s
        for si, xi in zip(support, x_s):
            coeffs[si] = np.abs(xi)

    sparse_db = 20 * np.log10(coeffs / (np.max(coeffs) + 1e-12) + 1e-12)
    return r_atoms, sparse_db

def peak_snr(pdb, r_axis, target_r, window_m=30):
    """
    Extract peak SNR near target range vs noise floor.
    Safe version with bounds checking.
    """

    idx_peak = np.argmin(np.abs(r_axis - target_r))

    # Prevent division issues
    spacing = max(abs(r_axis[1] - r_axis[0]), 1e-6)

    # Safe window size
    win = max(1, int(window_m / spacing))

    # Safe slice bounds
    start_idx = max(0, idx_peak - win)
    end_idx = min(len(pdb), idx_peak + win)

    region = pdb[start_idx:end_idx]

    # Prevent empty array crash
    if len(region) == 0:
        return 0

    peak = np.max(region)

    # Noise floor region
    noise_region = pdb[int(len(pdb) * 0.7):]

    if len(noise_region) == 0:
        noise_floor = 0
    else:
        noise_floor = np.percentile(noise_region, 50)

    return peak - noise_floor


# ============================================================
# PLOT 1  —  Zero-Padding: Resolution vs Padding Factor
# ============================================================
print("\n[1/8] Generating Plot 1: Zero-padding study...")

TARGET_R = 250.0
SNR1     = 18
beat1    = generate_beat_signal(TARGET_R, snr_db=SNR1)

pad_factors = [1, 2, 4, 8, 16]
colors1     = ['gray', 'steelblue', 'tomato', 'seagreen', 'purple']
labels1     = [f'{p}× padding (N_FFT = {N_samples * p})' for p in pad_factors]

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 9))
fig1.suptitle(f'Plot 1 — Zero-Padding FFT: Resolution vs Padding Factor\n'
              f'Target @ {TARGET_R} m  |  Phase 2 default = 4× padding',
              fontweight='bold')

for pf, col, lbl in zip(pad_factors, colors1, labels1):
    r_p, _, pdb_p = range_fft_zeropad(beat1.copy(), pad_factor=pf)
    peak_r_p = r_p[np.argmax(pdb_p)]
    lw = 2.0 if pf == 4 else 1.2
    ls = '--' if pf == 4 else '-'
    axes1[0].plot(r_p, pdb_p, color=col, lw=lw, ls=ls,
                  label=f'{lbl}  → peak @ {peak_r_p:.1f} m')
axes1[0].axvline(TARGET_R, color='black', ls=':', lw=2, label=f'True target: {TARGET_R} m')
axes1[0].set_xlim(0, max_range / 2)
axes1[0].set_xlabel('Range (m)')
axes1[0].set_ylabel('Power (dB)')
axes1[0].set_title('Full Range Profile — all padding factors')
axes1[0].legend(fontsize=8.5)

# Zoomed
for pf, col, lbl in zip(pad_factors, colors1, labels1):
    r_p, _, pdb_p = range_fft_zeropad(beat1.copy(), pad_factor=pf)
    lw = 2.0 if pf == 4 else 1.2
    ls = '--' if pf == 4 else '-'
    axes1[1].plot(r_p, pdb_p, color=col, lw=lw, ls=ls, label=lbl)
axes1[1].axvline(TARGET_R, color='black', ls=':', lw=2, label='True target')
axes1[1].set_xlim(TARGET_R - 15, TARGET_R + 15)
axes1[1].set_xlabel('Range (m)')
axes1[1].set_ylabel('Power (dB)')
axes1[1].set_title('Zoomed: Peak interpolation improves with padding\n'
                   '(Does NOT improve true range resolution — only peak localisation)')
axes1[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot1_zero_padding_study.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot1_zero_padding_study.png")


# ============================================================
# PLOT 2  —  Windowing + Zero-Padding Combined
# ============================================================
print("[2/8] Generating Plot 2: Windowing + zero-padding combined...")

TARGET_R2 = 300.0
SNR2      = 15
beat2     = generate_beat_signal(TARGET_R2, snr_db=SNR2)

configs2 = [
    ('No window,  1× pad  (worst)',   np.ones(N_samples),     1,  'gray'),
    ('Hanning,    4× pad  [P2 base]', np.hanning(N_samples),  4,  'steelblue'),
    ('Hanning,   16× pad  (enhanced)',np.hanning(N_samples),  16, 'tomato'),
    ('Blackman,  16× pad  (enhanced)',np.blackman(N_samples),  16, 'seagreen'),
]

NFFT2_max = N_samples * 16
r_ref     = np.linspace(0, max_range, NFFT2_max // 2)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle(f'Plot 2 — Windowing + Zero-Padding Combined Effect\n'
              f'Target @ {TARGET_R2} m  |  SNR = {SNR2} dB',
              fontweight='bold')

snr_gains2 = []
for cfg_name, win, pad, col in configs2:
    windowed = beat2.copy() * win
    n_fft2   = N_samples * pad
    fft2     = np.abs(np.fft.fft(windowed, n=n_fft2)[:n_fft2 // 2])
    r2       = np.linspace(0, max_range, n_fft2 // 2)
    pdb2     = 20 * np.log10(fft2 + 1e-12)
    peak_r2  = r2[np.argmax(pdb2)]
    snr2     = peak_snr(pdb2, r2, TARGET_R2)
    snr_gains2.append(snr2)
    lw = 2.0 if 'P2' in cfg_name else 1.2
    ls = '--' if 'P2' in cfg_name else '-'
    axes2[0].plot(r2, pdb2, color=col, lw=lw, ls=ls,
                  label=f'{cfg_name} → SNR = {snr2:.1f} dB')
axes2[0].axvline(TARGET_R2, color='black', ls=':', lw=2)
axes2[0].set_xlim(TARGET_R2 - 25, TARGET_R2 + 25)
axes2[0].set_xlabel('Range (m)')
axes2[0].set_ylabel('Power (dB)')
axes2[0].set_title('Zoomed: Peak sharpness comparison')
axes2[0].legend(fontsize=8)

cfg_names2 = [c[0].split('(')[0].strip() for c in configs2]
bars2 = axes2[1].bar(range(len(configs2)), snr_gains2,
                     color=[c[3] for c in configs2], alpha=0.85, width=0.6)
axes2[1].set_xticks(range(len(configs2)))
axes2[1].set_xticklabels(cfg_names2, rotation=15, ha='right', fontsize=8.5)
axes2[1].set_ylabel('Peak SNR (dB)')
axes2[1].set_title('Peak SNR Comparison\n(Higher = better enhancement)')
for bar, v in zip(bars2, snr_gains2):
    axes2[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3, f'{v:.1f}',
                  ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot2_windowing_zeropadding.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot2_windowing_zeropadding.png")


# ============================================================
# PLOT 3  —  MUSIC Super-Resolution
# ============================================================
print("[3/8] Generating Plot 3: MUSIC super-resolution...")

# Two closely spaced targets — 1.5× range_res apart (hard to separate with FFT)
R3_a = 200.0
R3_b = R3_a + 1.5 * range_res   # barely separated
SNR3 = 20

beat3 = generate_multi_target_beat([R3_a, R3_b], [SNR3, SNR3 - 3])

# Baseline FFT (Phase 2)
r_fft3, _, pdb_fft3 = range_fft(beat3.copy())
# Enhanced FFT (16× padding)
r_fft3e, _, pdb_fft3e = range_fft_zeropad(beat3.copy(), pad_factor=16)
# MUSIC
r_music3, pdb_music3 = music_pseudospectrum(beat3.copy(), n_sources=2, n_range_bins=2000)

fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5.5))
fig3.suptitle(f'Plot 3 — MUSIC Super-Resolution vs Standard FFT\n'
              f'Two targets at {R3_a:.0f} m & {R3_b:.1f} m  (separation = {R3_b - R3_a:.1f} m = 1.5× Δr = {range_res:.2f} m)',
              fontweight='bold')

for ax, r_ax, pdb, title, col in [
    (axes3[0], r_fft3,   pdb_fft3,   f'Phase 2 Baseline FFT\n(4× padding)', 'steelblue'),
    (axes3[1], r_fft3e,  pdb_fft3e,  f'Enhanced FFT\n(16× padding)',         'tomato'),
    (axes3[2], r_music3, pdb_music3, f'MUSIC Super-Resolution\n(2 sources)',   'seagreen'),
]:
    ax.plot(r_ax, pdb, color=col, lw=1.2)
    for R_m, lab in [(R3_a, 'T1'), (R3_b, 'T2')]:
        ax.axvline(R_m, color='black', ls=':', lw=1.5, label=f'{lab} @ {R_m:.1f} m')
    ax.set_xlim(R3_a - 10, R3_b + 10)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(title)
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot3_music_superresolution.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot3_music_superresolution.png")


# ============================================================
# PLOT 4  —  OS-CFAR vs CA-CFAR in Clutter
# ============================================================
print("[4/8] Generating Plot 4: OS-CFAR vs CA-CFAR...")

# Scenario: strong clutter target + weak target nearby
R4_clutter = 200.0; SNR4_clutter = 30
R4_weak    = 250.0; SNR4_weak    = 8
SNR4       = 8

beat4 = generate_multi_target_beat(
    [R4_clutter, R4_weak],
    [SNR4_clutter, SNR4_weak]
)
r4, _, pdb4 = range_fft(beat4)
thr_ca4, det_ca4 = ca_cfar_1d(pdb4, Tr=12, Gr=5, offset_db=8)
thr_os4, det_os4 = os_cfar_1d(pdb4, Tr=12, Gr=5, offset_db=8, k_rank=0.75)

fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.5))
fig4.suptitle(f'Plot 4 — OS-CFAR vs CA-CFAR in Clutter Environment\n'
              f'Strong clutter @ {R4_clutter} m (SNR={SNR4_clutter} dB)  |  Weak target @ {R4_weak} m (SNR={SNR4_weak} dB)',
              fontweight='bold')

for ax, thr, det, title, cfar_col in [
    (axes4[0], thr_ca4, det_ca4, 'CA-CFAR (Phase 2 method)\n— masking by strong target', 'steelblue'),
    (axes4[1], thr_os4, det_os4, 'OS-CFAR (Enhanced)\n— robust to target masking',        'tomato'),
]:
    ax.plot(r4, pdb4, 'gray', lw=1, label='Range profile', alpha=0.8)
    ax.plot(r4, thr,  color=cfar_col, lw=1.5, ls='--', label='CFAR threshold')
    det_r = r4[det]
    if len(det_r) > 0:
        ax.scatter(det_r, pdb4[det], color='lime', s=60, zorder=5,
                   edgecolors='black', label=f'{det.sum()} detections')
    ax.axvline(R4_clutter, color='orange', ls=':', lw=1.5, label=f'Clutter @ {R4_clutter} m')
    ax.axvline(R4_weak,    color='red',    ls=':', lw=1.5, label=f'Weak target @ {R4_weak} m')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(100, 400)

plt.tight_layout()
plt.savefig('plot4_os_cfar_comparison.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot4_os_cfar_comparison.png")


# ============================================================
# PLOT 5  —  Coherent Signal Integration: SNR vs Chirps
# ============================================================
print("[5/8] Generating Plot 5: Coherent integration SNR gain...")

TARGET_R5 = 400.0
SNR5_raw  = 5    # very low — single chirp barely detectable
n_int_vals = [1, 2, 4, 8, 16, 32, 64, 128]
colors5   = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_int_vals)))

snr_measured = []
snr_theory   = []

for n_int in n_int_vals:
    r5, pdb5 = coherent_integration(TARGET_R5, n_int, snr_db=SNR5_raw)
    snr5 = peak_snr(pdb5, r5, TARGET_R5)
    snr_measured.append(snr5)
    snr_theory.append(SNR5_raw + 10 * np.log10(n_int))

fig5, axes5 = plt.subplots(1, 3, figsize=(17, 5.5))
fig5.suptitle(f'Plot 5 — Coherent Signal Integration: SNR vs Number of Integrated Chirps\n'
              f'Target @ {TARGET_R5} m  |  Single-chirp SNR = {SNR5_raw} dB',
              fontweight='bold')

# 5a: range profiles for selected n_int
for n_int, col in [(1, 'tomato'), (8, 'orange'), (64, 'seagreen'), (128, 'steelblue')]:
    r5, pdb5 = coherent_integration(TARGET_R5, n_int, snr_db=SNR5_raw)
    axes5[0].plot(r5, pdb5, lw=1.2, color=col, label=f'N={n_int} chirps')
axes5[0].axvline(TARGET_R5, color='black', ls=':', lw=1.8)
axes5[0].set_xlabel('Range (m)')
axes5[0].set_ylabel('Power (dB)')
axes5[0].set_title('Range Profiles: N=1 vs N=128')
axes5[0].set_xlim(TARGET_R5 - 60, TARGET_R5 + 60)
axes5[0].legend(fontsize=8.5)
axes5[0].axhline(SNR_THRESHOLD_DB, color='red', ls='--', lw=1.2, alpha=0.6,
                 label=f'Threshold {SNR_THRESHOLD_DB} dB')

# 5b: SNR vs n_int (theory vs measured)
axes5[1].semilogx(n_int_vals, snr_measured, 'b-o', ms=7, lw=2, label='Measured SNR')
axes5[1].semilogx(n_int_vals, snr_theory,   'r--s', ms=7, lw=1.5, label='Theory: +10log10(N)')
axes5[1].axhline(SNR_THRESHOLD_DB, color='black', ls=':', lw=1.5, label='Detection threshold')
axes5[1].set_xlabel('Number of integrated chirps (N)')
axes5[1].set_ylabel('Peak SNR (dB)')
axes5[1].set_title('SNR Gain vs Chirps Integrated\n(Theory confirms 10 dB/decade)')
axes5[1].legend(fontsize=8.5)

# 5c: SNR gain relative to single chirp
snr_gain5 = np.array(snr_measured) - snr_measured[0]
theory_gain5 = 10 * np.log10(np.array(n_int_vals))
axes5[2].plot(n_int_vals, snr_gain5,    'b-o', ms=7, lw=2, label='Measured gain')
axes5[2].plot(n_int_vals, theory_gain5, 'r--', lw=1.5, label='Theory: 10log10(N)')
axes5[2].set_xlabel('Number of chirps')
axes5[2].set_ylabel('SNR gain (dB) vs single chirp')
axes5[2].set_title('SNR Gain from Coherent Integration')
axes5[2].legend()

plt.tight_layout()
plt.savefig('plot5_coherent_integration.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot5_coherent_integration.png")


# ============================================================
# PLOT 6  —  Sparse / OMP Reconstruction
# ============================================================
print("[6/8] Generating Plot 6: Sparse OMP reconstruction...")

R6_targets = [180.0, 182.0, 300.0]    # first two are very close
SNR6       = 18
beat6 = generate_multi_target_beat(R6_targets, [SNR6, SNR6 - 5, SNR6 + 3])

# Baseline FFT
r6_fft, _, pdb6_fft = range_fft(beat6.copy(), n_fft=N_samples * 4)
# Enhanced FFT (16×)
r6_enh, _, pdb6_enh = range_fft_zeropad(beat6.copy(), pad_factor=16)
# OMP sparse
r6_omp, pdb6_omp = omp_sparse_reconstruction(beat6.copy(), n_range_atoms=800, sparsity=4)

fig6, axes6 = plt.subplots(1, 3, figsize=(17, 5.5))
fig6.suptitle(f'Plot 6 — Sparse / OMP Reconstruction\n'
              f'Targets @ {R6_targets} m  |  First two separated by {R6_targets[1]-R6_targets[0]:.1f} m (< Δr = {range_res:.2f} m!)',
              fontweight='bold')

for ax, r_ax, pdb, title, col in [
    (axes6[0], r6_fft, pdb6_fft, f'Phase 2 Baseline FFT\n(4× padding)', 'steelblue'),
    (axes6[1], r6_enh, pdb6_enh, f'Enhanced FFT (16× padding)',           'tomato'),
    (axes6[2], r6_omp, pdb6_omp, f'OMP Sparse Reconstruction\n(sparsity k=4)', 'seagreen'),
]:
    ax.plot(r_ax, pdb, color=col, lw=1.2)
    for R_m in R6_targets:
        ax.axvline(R_m, color='black', ls=':', lw=1.5, alpha=0.8)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Power (dB) — normalised')
    ax.set_title(title)
    # zoom to close pair
    ax.set_xlim(R6_targets[0] - 8, R6_targets[1] + 8)

plt.tight_layout()
plt.savefig('plot6_sparse_reconstruction.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot6_sparse_reconstruction.png")


# ============================================================
# PLOT 7  —  Algorithm Benchmark: Pd and Peak Sharpness
# ============================================================
print("[7/8] Generating Plot 7: Algorithm benchmark...")

TARGET_R7 = 350.0
SNR_VALS7 = np.arange(2, 20, 2)    # SNR sweep
N_TRIALS7 = 30

methods7 = {
    'Baseline FFT (4×)': lambda b: range_fft(b, n_fft=N_samples * 4),
    'Enhanced FFT (16×)': lambda b: range_fft_zeropad(b, pad_factor=16),
}

pd_results = {m: [] for m in methods7}
cfar_methods = {
    'CA-CFAR (P2 baseline)': ca_cfar_1d,
    'OS-CFAR (enhanced)':    lambda p, **kw: os_cfar_1d(p, k_rank=0.75, **kw),
}
pd_cfar = {m: [] for m in cfar_methods}

for snr in SNR_VALS7:
    for method_name, fft_fn in methods7.items():
        pd_sum = 0
        for _ in range(N_TRIALS7):
            beat7 = generate_beat_signal(TARGET_R7, snr_db=snr)
            r7, _, pdb7 = fft_fn(beat7)
            peak_r7 = r7[np.argmax(pdb7)]
            if abs(peak_r7 - TARGET_R7) < 30:
                pd_sum += 1
        pd_results[method_name].append(pd_sum / N_TRIALS7)

    for cfar_name, cfar_fn in cfar_methods.items():
        pd_sum = 0
        for _ in range(N_TRIALS7):
            beat7 = generate_beat_signal(TARGET_R7, snr_db=snr)
            r7, _, pdb7 = range_fft(beat7)
            thr7, det7 = cfar_fn(pdb7, Tr=10, Gr=4, offset_db=8)
            t_idx = np.argmin(np.abs(r7 - TARGET_R7))
            detected = np.any(det7[max(0, t_idx - 15): t_idx + 15])
            pd_sum += int(detected)
        pd_cfar[cfar_name].append(pd_sum / N_TRIALS7)

fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5.5))
fig7.suptitle(f'Plot 7 — Algorithm Benchmark: Detection Probability\n'
              f'Target @ {TARGET_R7} m  |  {N_TRIALS7} Monte Carlo trials per SNR point',
              fontweight='bold')

colors_m7 = ['steelblue', 'tomato']
for (m_name, pd_vals), col in zip(pd_results.items(), colors_m7):
    ls = '--' if 'Baseline' in m_name else '-'
    axes7[0].plot(SNR_VALS7, pd_vals, lw=2, ls=ls, color=col,
                  marker='o', ms=6, label=m_name)
axes7[0].axhline(0.9, color='black', ls=':', lw=1.5, label='Pd = 0.9 target')
axes7[0].set_xlabel('SNR (dB)')
axes7[0].set_ylabel('Probability of Detection (Pd)')
axes7[0].set_title('Pd vs SNR: Baseline vs Enhanced FFT')
axes7[0].set_ylim(0, 1.05)
axes7[0].legend(fontsize=8.5)

colors_c7 = ['steelblue', 'tomato']
for (c_name, pd_vals), col in zip(pd_cfar.items(), colors_c7):
    ls = '--' if 'P2' in c_name else '-'
    axes7[1].plot(SNR_VALS7, pd_vals, lw=2, ls=ls, color=col,
                  marker='s', ms=6, label=c_name)
axes7[1].axhline(0.9, color='black', ls=':', lw=1.5, label='Pd = 0.9 target')
axes7[1].set_xlabel('SNR (dB)')
axes7[1].set_ylabel('Probability of Detection (Pd)')
axes7[1].set_title('Pd vs SNR: CA-CFAR vs OS-CFAR')
axes7[1].set_ylim(0, 1.05)
axes7[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot7_algorithm_benchmark.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot7_algorithm_benchmark.png")


# ============================================================
# PLOT 8  —  SNR Gain Summary: All Methods vs Phase 2 Baseline
# ============================================================
print("[8/8] Generating Plot 8: SNR gain summary...")

TARGET_R8 = 300.0
SNR8      = 12    # low SNR to make differences visible
N_RUNS8   = 20

beat8_runs = [generate_beat_signal(TARGET_R8, snr_db=SNR8) for _ in range(N_RUNS8)]

def avg_peak_snr(beats, fft_fn):
    snrs = []
    for b in beats:
        r, _, pdb = fft_fn(b.copy())
        snrs.append(peak_snr(pdb, r, TARGET_R8))
    return np.mean(snrs)

def avg_peak_snr_integration(target_r, snr_db, n_int, n_runs):
    snrs = []
    for _ in range(n_runs):
        r, pdb = coherent_integration(target_r, n_int, snr_db=snr_db)
        snrs.append(peak_snr(pdb, r, target_r))
    return np.mean(snrs)

methods_8 = [
    ('Phase 2 Baseline\n(Hanning + 4× pad)',  avg_peak_snr(beat8_runs, lambda b: range_fft(b, n_fft=N_samples * 4))),
    ('Enhanced FFT\n(Hanning + 16× pad)',      avg_peak_snr(beat8_runs, lambda b: range_fft_zeropad(b, pad_factor=16))),
    ('Blackman + 16× pad',                     avg_peak_snr(beat8_runs,
        lambda b: (lambda bm: (np.linspace(0, max_range, N_samples * 8),
                               np.zeros(N_samples * 8),
                               20 * np.log10(np.abs(np.fft.fft(b * np.blackman(N_samples),
                                n=N_samples * 16)[:N_samples * 8]) + 1e-12)))(b))),
    ('Coherent\nIntegration N=32',             avg_peak_snr_integration(TARGET_R8, SNR8, 32, N_RUNS8)),
    ('Coherent\nIntegration N=128',            avg_peak_snr_integration(TARGET_R8, SNR8, 128, N_RUNS8)),
]

# Simpler recompute for Blackman
def blackman_enhanced(b):
    win = np.blackman(N_samples)
    n_f = N_samples * 16
    fft_b = np.abs(np.fft.fft(b * win, n=n_f)[:n_f // 2])
    r_b   = np.linspace(0, max_range, n_f // 2)
    pdb_b = 20 * np.log10(fft_b + 1e-12)
    return r_b, None, pdb_b

methods_8[2] = ('Blackman + 16× pad', avg_peak_snr(beat8_runs, blackman_enhanced))

baseline_snr = methods_8[0][1]
methods_labels = [m[0] for m in methods_8]
methods_snrs   = [m[1] for m in methods_8]
methods_gains  = [s - baseline_snr for s in methods_snrs]

colors8 = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']

fig8, axes8 = plt.subplots(1, 2, figsize=(15, 6))
fig8.suptitle(f'Plot 8 — SNR Gain Summary: All Enhancement Methods vs Phase 2 Baseline\n'
              f'Target @ {TARGET_R8} m  |  Input SNR = {SNR8} dB  |  {N_RUNS8} Monte Carlo runs',
              fontweight='bold')

bars8a = axes8[0].bar(range(len(methods_8)), methods_snrs, color=colors8, alpha=0.85, width=0.6)
axes8[0].axhline(baseline_snr, color='steelblue', ls='--', lw=2, label=f'Baseline: {baseline_snr:.1f} dB')
axes8[0].axhline(SNR_THRESHOLD_DB, color='red', ls=':', lw=1.5, label=f'Detection threshold: {SNR_THRESHOLD_DB} dB')
axes8[0].set_xticks(range(len(methods_8)))
axes8[0].set_xticklabels(methods_labels, rotation=10, ha='right', fontsize=8.5)
axes8[0].set_ylabel('Peak SNR (dB)')
axes8[0].set_title('Absolute Peak SNR per Method')
for bar, v in zip(bars8a, methods_snrs):
    axes8[0].text(bar.get_x() + bar.get_width() / 2, v + 0.2, f'{v:.1f}',
                  ha='center', fontsize=9, fontweight='bold')
axes8[0].legend(fontsize=8.5)

bars8b = axes8[1].bar(range(len(methods_8)), methods_gains, color=colors8, alpha=0.85, width=0.6)
axes8[1].axhline(0, color='black', ls='--', lw=1.5, label='Baseline (0 dB gain)')
axes8[1].set_xticks(range(len(methods_8)))
axes8[1].set_xticklabels(methods_labels, rotation=10, ha='right', fontsize=8.5)
axes8[1].set_ylabel('SNR Gain vs Baseline (dB)')
axes8[1].set_title('SNR Improvement over Phase 2 Baseline\n(Phase 3B proves enhancement)')
for bar, v in zip(bars8b, methods_gains):
    color = 'green' if v >= 0 else 'red'
    axes8[1].text(bar.get_x() + bar.get_width() / 2,
                  v + (0.1 if v >= 0 else -0.4),
                  f'{v:+.1f} dB', ha='center', fontsize=9,
                  fontweight='bold', color=color)
axes8[1].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot8_snr_gain_summary.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot8_snr_gain_summary.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print(DIVIDER)
print("  PHASE 3B RESULTS SUMMARY — RANGE ENHANCEMENT ALGORITHMS")
print(DIVIDER)
print(f"  {'Method':<35} {'Peak SNR':>10}  {'Gain':>10}")
print("-" * 60)
for m_name, m_snr, m_gain in zip(methods_labels, methods_snrs, methods_gains):
    m_name_flat = m_name.replace('\n', ' ')
    print(f"  {m_name_flat:<35} {m_snr:>8.1f} dB  {m_gain:>+8.1f} dB")
print(DIVIDER)
print()
print("  Figures saved:")
figs = [
    "plot1_zero_padding_study.png    — Zero-padding vs resolution",
    "plot2_windowing_zeropadding.png — Window + padding combined",
    "plot3_music_superresolution.png — MUSIC vs FFT close targets",
    "plot4_os_cfar_comparison.png    — OS-CFAR vs CA-CFAR",
    "plot5_coherent_integration.png  — Integration SNR gain",
    "plot6_sparse_reconstruction.png — OMP sparse recovery",
    "plot7_algorithm_benchmark.png   — Pd vs SNR (Monte Carlo)",
    "plot8_snr_gain_summary.png      — All methods vs baseline",
]
for i, name in enumerate(figs, 1):
    print(f"  {i}. {name}")
print()
print("  Done. Phase 3B complete — ready for Phase 4 combined pipeline.")
print(DIVIDER)

plt.show()