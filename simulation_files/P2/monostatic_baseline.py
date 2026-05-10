"""
FMCW Radar Simulation - Phase 2: Monostatic Baseline
=====================================================
Shows ALL Phase 2 results:
  Plot 1  - Raw beat signal in time domain
  Plot 2  - Chirp spectrogram (frequency vs time)
  Plot 3  - Range FFT: single target at different SNR levels
  Plot 4  - Range FFT: multiple targets + CFAR detection
  Plot 5  - Range-Doppler map (2D FFT heatmap)
  Plot 6  - Doppler profile: velocity extraction
  Plot 7  - CFAR threshold sweep: Pd vs threshold offset
  Plot 8  - SNR vs Range curve (monostatic radar equation)

Run:  python phase2_monostatic_baseline.py
Output: 8 PNG figures saved to current folder
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# SYSTEM PARAMETERS  (77 GHz automotive FMCW)
# ============================================================
c         = 3e8          # speed of light (m/s)
fc        = 77e9         # carrier frequency (Hz)
B         = 150e6        # chirp bandwidth (Hz)
Tm        = 5.5e-3       # chirp duration (s)
S         = B / Tm       # chirp slope (Hz/s)
lambda_   = c / fc       # wavelength ~3.9 mm

N_samples = 1024         # ADC samples per chirp
N_chirps  = 128          # chirps per CPI frame
fs        = N_samples / Tm  # ADC sampling rate

range_res    = c / (2 * B)        # range resolution  ~1 m
max_range    = c * Tm / 2         # max unambiguous range ~825 m
vel_res      = lambda_ / (2 * N_chirps * Tm)  # velocity resolution
max_vel      = lambda_ / (4 * Tm)             # max unambiguous velocity

SNR_THRESHOLD_DB = 13   # minimum detectable SNR

DIVIDER = "=" * 60

print(DIVIDER)
print("  PHASE 2 — MONOSTATIC FMCW BASELINE SIMULATION")
print(DIVIDER)
print(f"  Carrier frequency   : {fc/1e9:.0f} GHz")
print(f"  Bandwidth           : {B/1e6:.0f} MHz")
print(f"  Chirp duration      : {Tm*1e3:.1f} ms")
print(f"  Chirp slope         : {S/1e12:.2f} THz/s")
print(f"  Wavelength          : {lambda_*1e3:.2f} mm")
print(f"  ADC samples/chirp   : {N_samples}")
print(f"  Chirps per frame    : {N_chirps}")
print(DIVIDER)
print(f"  Range resolution    : {range_res:.2f} m")
print(f"  Max unambiguous range: {max_range:.1f} m")
print(f"  Velocity resolution : {vel_res:.3f} m/s")
print(f"  Max velocity        : {max_vel:.2f} m/s")
print(f"  Detection threshold : {SNR_THRESHOLD_DB} dB")
print(DIVIDER)


# ============================================================
# CORE DSP FUNCTIONS
# ============================================================

def make_time_axis():
    return np.linspace(0, Tm, N_samples)

def beat_freq_from_range(R):
    """Beat frequency for a target at range R."""
    return S * (2 * R / c)

def generate_beat_signal(target_range, snr_db=20):
    """
    Generate FMCW beat signal for a single point target.
    Returns real-valued beat signal with additive white Gaussian noise.
    """
    t   = make_time_axis()
    tau = 2 * target_range / c
    fb  = S * tau
    sig = np.cos(2 * np.pi * fb * t)
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(N_samples)
    return sig + noise

def generate_multi_target_beat(targets_ranges, targets_snr):
    """Generate beat signal for multiple targets."""
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
    """
    Apply window + FFT on beat signal.
    Returns (range_axis_m, power_linear, power_dB).
    """
    if n_fft is None:
        n_fft = N_samples * 4      # 4x zero-padding by default
    if window:
        win = np.hanning(len(beat_sig))
        beat_sig = beat_sig * win
    fft_out  = np.fft.fft(beat_sig, n=n_fft)
    fft_mag  = np.abs(fft_out[:n_fft // 2])
    r_axis   = np.linspace(0, max_range, n_fft // 2)
    pdb      = 20 * np.log10(fft_mag + 1e-12)
    return r_axis, fft_mag, pdb

def ca_cfar_1d(power_db, Tr=10, Gr=4, offset_db=8):
    """
    Cell-Averaging CFAR on 1D range profile.
    Tr        : training cells (each side)
    Gr        : guard cells (each side)
    offset_db : CFAR offset above estimated noise floor
    Returns   : threshold array (dB), binary detection mask
    """
    N         = len(power_db)
    threshold = np.full(N, np.nan)
    detected  = np.zeros(N, dtype=bool)
    for i in range(Tr + Gr, N - Tr - Gr):
        left       = power_db[i - Tr - Gr : i - Gr]
        right      = power_db[i + Gr + 1  : i + Gr + Tr + 1]
        noise_est  = np.mean(np.concatenate([left, right]))
        threshold[i] = noise_est + offset_db
        detected[i]  = power_db[i] > threshold[i]
    return threshold, detected

def build_range_doppler(R_targets, vel_targets, snr_db=18):
    """
    Build full Range-Doppler matrix using 2D FFT.
    Slow-time axis: chirp index  → Doppler / velocity
    Fast-time axis: sample index → range
    Returns: (rd_map magnitude, range_axis, velocity_axis)
    """
    rd = np.zeros((N_chirps, N_samples), dtype=complex)
    t  = make_time_axis()
    for ci in range(N_chirps):
        sig = np.zeros(N_samples)
        for R, v in zip(R_targets, vel_targets):
            R_now = R + v * ci * Tm
            tau   = 2 * R_now / c
            fb    = S * tau
            fd    = 2 * v * fc / c
            sig  += np.cos(2 * np.pi * (fb + fd) * t)
        noise_amp = 10 ** (-snr_db / 20)
        sig += noise_amp * np.random.randn(N_samples)
        # Range FFT per chirp (fast-time)
        win = np.hanning(N_samples)
        rd[ci, :] = np.fft.fft(sig * win, N_samples)

    # Doppler FFT across chirps (slow-time), apply Hanning window
    slow_win = np.hanning(N_chirps).reshape(-1, 1)
    rd_map   = np.fft.fftshift(np.fft.fft(rd * slow_win, N_chirps, axis=0), axes=0)

    r_axis   = np.linspace(0, max_range / 2, N_samples // 2)
    v_axis   = np.linspace(-max_vel, max_vel, N_chirps)
    return np.abs(rd_map[:, :N_samples // 2]), r_axis, v_axis

def snr_vs_range(R_array, snr_ref_db=65, R_ref=50):
    """Monostatic SNR (dB): drops as 1/R^4."""
    return snr_ref_db - 40 * np.log10(np.maximum(R_array, 1e-3) / R_ref)

def max_detection_range(snr_db_curve, r_axis):
    idx = np.where(snr_db_curve >= SNR_THRESHOLD_DB)[0]
    return r_axis[idx[-1]] if len(idx) > 0 else 0.0


# ============================================================
# PLOT 1  —  Raw Beat Signal (Time Domain)
# ============================================================
print("\n[1/8] Generating Plot 1: Raw beat signal...")

TARGET_R   = 150.0   # target at 150 m
BEAT_SNR   = 25      # dB
beat_clean = generate_beat_signal(TARGET_R, snr_db=100)   # near-noiseless
beat_noisy = generate_beat_signal(TARGET_R, snr_db=BEAT_SNR)
t_us       = make_time_axis() * 1e6                        # microseconds
fb_true    = beat_freq_from_range(TARGET_R)

fig1, axes = plt.subplots(3, 1, figsize=(14, 9))
fig1.suptitle('Plot 1 — Raw FMCW Beat Signal (Time Domain)\n'
              f'Target @ {TARGET_R} m  |  Beat frequency = {fb_true/1e3:.1f} kHz',
              fontweight='bold')

# 1a: clean signal
axes[0].plot(t_us, beat_clean, color='steelblue', lw=0.8)
axes[0].set_title('Clean beat signal (no noise)')
axes[0].set_ylabel('Amplitude')
axes[0].set_xlim(0, t_us[-1])

# 1b: noisy signal
axes[1].plot(t_us, beat_noisy, color='tomato', lw=0.6, alpha=0.85)
axes[1].set_title(f'Noisy beat signal (SNR = {BEAT_SNR} dB)')
axes[1].set_ylabel('Amplitude')
axes[1].set_xlim(0, t_us[-1])

# 1c: zoom on first 0.2 ms to see oscillations clearly
zoom_mask = t_us < 200
axes[2].plot(t_us[zoom_mask], beat_clean[zoom_mask], 'steelblue', lw=1.2, label='Clean')
axes[2].plot(t_us[zoom_mask], beat_noisy[zoom_mask], 'tomato',    lw=0.8, alpha=0.7, label='Noisy')
axes[2].set_title('Zoomed view (first 0.2 ms) — beat oscillation visible')
axes[2].set_xlabel('Time (µs)')
axes[2].set_ylabel('Amplitude')
axes[2].legend()
axes[2].set_xlim(0, 200)

for ax in axes:
    ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('plot1_beat_signal.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot1_beat_signal.png")


# ============================================================
# PLOT 2  —  Chirp Spectrogram
# ============================================================
print("[2/8] Generating Plot 2: Chirp spectrogram...")

beat_multi = generate_multi_target_beat([80, 200, 400], [25, 20, 15])

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Plot 2 — Chirp Spectrogram  |  Frequency Content vs Time',
              fontweight='bold')

for ax, (sig, label, col) in zip(axes2, [
    (beat_clean, f'Single target @ {TARGET_R} m', 'Blues'),
    (beat_multi, 'Three targets @ 80, 200, 400 m',  'Oranges'),
]):
    f_spect, t_spect, Sxx = spectrogram(sig, fs=fs, nperseg=64, noverlap=48)
    ax.pcolormesh(t_spect * 1e6, f_spect / 1e3, 10 * np.log10(Sxx + 1e-12),
                  cmap=col, shading='gouraud')
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(label)
    ax.set_ylim(0, 200)

plt.tight_layout()
plt.savefig('plot2_spectrogram.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot2_spectrogram.png")


# ============================================================
# PLOT 3  —  Range FFT: SNR Effect
# ============================================================
print("[3/8] Generating Plot 3: Range FFT at different SNR levels...")

snr_levels = [30, 20, 15, 10, 7]
TARGET_R3  = 250.0
colors3    = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))
fig3.suptitle(f'Plot 3 — Range FFT: Effect of SNR  |  Target @ {TARGET_R3} m',
              fontweight='bold')

for snr, col in zip(snr_levels, colors3):
    sig        = generate_beat_signal(TARGET_R3, snr_db=snr)
    r_ax, _, pdb = range_fft(sig)
    axes3[0].plot(r_ax, pdb, color=col, lw=0.9, alpha=0.85, label=f'SNR = {snr} dB')

axes3[0].axvline(TARGET_R3, color='black', ls=':', lw=2, label=f'True: {TARGET_R3} m')
axes3[0].set_xlabel('Range (m)')
axes3[0].set_ylabel('Power (dBFS)')
axes3[0].set_title('Range profile at different SNR levels')
axes3[0].set_xlim(0, 700)
axes3[0].set_ylim(-40, 100)
axes3[0].legend(fontsize=9)

# Right subplot: zoom around target ± 50 m
zoom_lo, zoom_hi = TARGET_R3 - 60, TARGET_R3 + 60
for snr, col in zip(snr_levels, colors3):
    sig        = generate_beat_signal(TARGET_R3, snr_db=snr)
    r_ax, _, pdb = range_fft(sig)
    mask = (r_ax >= zoom_lo) & (r_ax <= zoom_hi)
    axes3[1].plot(r_ax[mask], pdb[mask], color=col, lw=1.2, label=f'{snr} dB')

axes3[1].axvline(TARGET_R3, color='black', ls=':', lw=2, label=f'True: {TARGET_R3} m')
axes3[1].set_xlabel('Range (m)')
axes3[1].set_ylabel('Power (dBFS)')
axes3[1].set_title(f'Zoomed: ±60 m around target')
axes3[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot3_range_fft_snr.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot3_range_fft_snr.png")


# ============================================================
# PLOT 4  —  Range FFT: Multiple Targets + CFAR Detection
# ============================================================
print("[4/8] Generating Plot 4: Multiple targets + CFAR...")

targets_R   = [80, 180, 310, 500]
targets_snr = [28, 22,  18,  12]

beat_mt     = generate_multi_target_beat(targets_R, targets_snr)
r_ax4, _, pdb4 = range_fft(beat_mt)
thr4, det4  = ca_cfar_1d(pdb4, Tr=12, Gr=5, offset_db=8)

fig4, axes4 = plt.subplots(2, 1, figsize=(14, 8))
fig4.suptitle('Plot 4 — Range FFT: Multi-Target Detection + CA-CFAR\n'
              'Targets @ 80 m, 180 m, 310 m, 500 m',
              fontweight='bold')

# Full range view
axes4[0].plot(r_ax4, pdb4, color='steelblue', lw=0.8, label='Range profile')
axes4[0].plot(r_ax4, thr4, color='orange',    lw=1.5, ls='--', label='CA-CFAR threshold')
axes4[0].scatter(r_ax4[det4], pdb4[det4], color='red', s=15, zorder=5, label='Detected')
for R in targets_R:
    axes4[0].axvline(R, color='lime', ls=':', lw=1.5)
axes4[0].axvline(targets_R[0], color='lime', ls=':', lw=1.5, label='True target positions')
axes4[0].set_xlabel('Range (m)')
axes4[0].set_ylabel('Power (dBFS)')
axes4[0].set_title('Full range profile — all 4 targets')
axes4[0].set_xlim(0, 700)
axes4[0].legend(fontsize=9)

# Annotate each detected target with estimated range
detected_ranges = r_ax4[det4]
# Group detections: peaks near each true target
for true_R in targets_R:
    near = detected_ranges[np.abs(detected_ranges - true_R) < 30]
    if len(near) > 0:
        est_R = near[np.argmax(pdb4[det4][np.abs(detected_ranges - true_R) < 30])]
        err   = abs(est_R - true_R)
        axes4[0].annotate(f'Est: {est_R:.0f}m\n(err {err:.1f}m)',
                          xy=(est_R, pdb4[np.argmin(np.abs(r_ax4 - est_R))]),
                          xytext=(est_R + 20, pdb4[np.argmin(np.abs(r_ax4 - est_R))] + 8),
                          arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                          fontsize=8, color='darkred')

# Error table below
axes4[1].axis('off')
table_data = []
for true_R in targets_R:
    near = detected_ranges[np.abs(detected_ranges - true_R) < 30]
    if len(near) > 0:
        est_R = near[np.argmax(pdb4[det4][np.abs(detected_ranges - true_R) < 30])]
        err   = est_R - true_R
        snr_v = targets_snr[targets_R.index(true_R)]
        fb    = beat_freq_from_range(true_R)
        table_data.append([f'{true_R} m', f'{est_R:.1f} m', f'{err:+.1f} m',
                           f'{snr_v} dB', f'{fb/1e3:.2f} kHz', 'DETECTED ✓'])
    else:
        table_data.append([f'{true_R} m', '—', '—',
                           f'{targets_snr[targets_R.index(true_R)]} dB', '—', 'MISSED ✗'])

col_labels = ['True range', 'Estimated', 'Error', 'SNR', 'Beat freq', 'Status']
tbl = axes4[1].table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, row in enumerate(table_data):
    bg = '#eaf4ea' if 'DETECTED' in row[-1] else '#fdecea'
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(bg)

axes4[1].set_title('Detection accuracy per target', pad=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plot4_multi_target_cfar.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot4_multi_target_cfar.png")


# ============================================================
# PLOT 5  —  Range-Doppler Map (2D FFT Heatmap)
# ============================================================
print("[5/8] Generating Plot 5: Range-Doppler map...")

RD_targets = [100, 280, 380]
RD_vels    = [15, -10, 5]      # m/s (positive = approaching)

rd_map, r_rd, v_rd = build_range_doppler(RD_targets, RD_vels)
rd_db = 20 * np.log10(rd_map + 1e-12)

fig5, ax5 = plt.subplots(figsize=(13, 6))
fig5.suptitle('Plot 5 — Range-Doppler Map (2D FFT Heatmap)\n'
              'Targets: (100m, +15m/s), (280m, -10m/s), (380m, +5m/s)',
              fontweight='bold')

im = ax5.imshow(rd_db, aspect='auto', origin='lower',
                extent=[r_rd[0], r_rd[-1], v_rd[0], v_rd[-1]],
                cmap='jet',
                vmin=np.percentile(rd_db, 60),
                vmax=np.percentile(rd_db, 99.5))
cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Power (dB)')

for R, v in zip(RD_targets, RD_vels):
    ax5.plot(R, v, 'w+', markersize=16, markeredgewidth=2.5)
    ax5.annotate(f'{R}m, {v:+}m/s',
                 xy=(R, v), xytext=(R + 12, v + 3),
                 color='white', fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1))

ax5.set_xlabel('Range (m)')
ax5.set_ylabel('Velocity (m/s)')
ax5.set_title('Range-Doppler heatmap — bright spots = targets\n'
              '(white + markers show true positions)')
ax5.axhline(0, color='white', lw=0.7, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot5_range_doppler_map.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot5_range_doppler_map.png")


# ============================================================
# PLOT 6  —  Doppler / Velocity Profile
# ============================================================
print("[6/8] Generating Plot 6: Velocity profile extraction...")

fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
fig6.suptitle('Plot 6 — Velocity Extraction from Doppler Profile\n'
              'Each subplot: range slice through one target → velocity profile',
              fontweight='bold')

for ax, (R_tgt, v_tgt, col) in zip(axes6, [
    (RD_targets[0], RD_vels[0], 'steelblue'),
    (RD_targets[1], RD_vels[1], 'tomato'),
    (RD_targets[2], RD_vels[2], 'seagreen'),
]):
    r_idx      = np.argmin(np.abs(r_rd - R_tgt))
    dop_slice  = rd_db[:, r_idx]
    ax.plot(v_rd, dop_slice, color=col, lw=1.2)
    peak_v     = v_rd[np.argmax(dop_slice)]
    ax.axvline(peak_v, color='black',  ls='--', lw=1.5, label=f'Detected: {peak_v:.1f} m/s')
    ax.axvline(v_tgt,  color='orange', ls=':',  lw=2.0, label=f'True: {v_tgt:+} m/s')
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'Target @ {R_tgt} m\nTrue vel = {v_tgt:+} m/s')
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig('plot6_velocity_profile.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot6_velocity_profile.png")


# ============================================================
# PLOT 7  —  CFAR Threshold Sweep
# ============================================================
print("[7/8] Generating Plot 7: CFAR threshold sweep analysis...")

TARGET_R7  = 300.0
SNR_DB7    = 14
offset_vals = np.arange(2, 20, 0.5)

n_trials    = 40
pd_list     = []
pfa_list    = []

for off in offset_vals:
    pd_sum  = 0
    pfa_sum = 0
    for _ in range(n_trials):
        sig       = generate_beat_signal(TARGET_R7, snr_db=SNR_DB7)
        r_ax7, _, pdb7 = range_fft(sig)
        thr7, det7 = ca_cfar_1d(pdb7, Tr=10, Gr=4, offset_db=off)
        # target bin index
        t_idx  = np.argmin(np.abs(r_ax7 - TARGET_R7))
        window = 8
        target_detected = np.any(det7[max(0, t_idx - window): t_idx + window])
        pd_sum += int(target_detected)
        # false alarms: detections far from target
        fa_mask = np.ones(len(det7), dtype=bool)
        fa_mask[max(0, t_idx - 20): t_idx + 20] = False
        pfa_sum += det7[fa_mask].sum() / fa_mask.sum()
    pd_list.append(pd_sum / n_trials)
    pfa_list.append(pfa_sum / n_trials)

fig7, axes7 = plt.subplots(1, 3, figsize=(15, 5))
fig7.suptitle(f'Plot 7 — CA-CFAR Analysis  |  Target @ {TARGET_R7} m, SNR = {SNR_DB7} dB',
              fontweight='bold')

axes7[0].plot(offset_vals, pd_list, 'b-o', markersize=4, lw=1.5)
axes7[0].axhline(0.9, color='red', ls='--', lw=1.2, label='Pd = 0.9 target')
axes7[0].set_xlabel('CFAR offset (dB)')
axes7[0].set_ylabel('Probability of detection (Pd)')
axes7[0].set_title('Pd vs CFAR offset')
axes7[0].set_ylim(0, 1.05)
axes7[0].legend()

axes7[1].plot(offset_vals, pfa_list, 'r-o', markersize=4, lw=1.5)
axes7[1].set_xlabel('CFAR offset (dB)')
axes7[1].set_ylabel('False alarm rate (Pfa)')
axes7[1].set_title('Pfa vs CFAR offset')
axes7[1].set_ylim(bottom=0)

axes7[2].plot(pfa_list, pd_list, 'g-o', markersize=4, lw=1.5)
axes7[2].set_xlabel('False alarm rate (Pfa)')
axes7[2].set_ylabel('Probability of detection (Pd)')
axes7[2].set_title('ROC Curve\n(Receiver Operating Characteristic)')
axes7[2].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4, label='Random guess')
axes7[2].set_xlim(0, 1)
axes7[2].set_ylim(0, 1.05)
axes7[2].legend()

plt.tight_layout()
plt.savefig('plot7_cfar_analysis.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot7_cfar_analysis.png")


# ============================================================
# PLOT 8  —  SNR vs Range (Radar Equation)
# ============================================================
print("[8/8] Generating Plot 8: SNR vs Range (Radar equation)...")

R_sweep = np.linspace(10, 800, 2000)
snr_curve = snr_vs_range(R_sweep)
R_max_detect = max_detection_range(snr_curve, R_sweep)

# Show effect of different transmit power / antenna gain assumptions
snr_low  = snr_vs_range(R_sweep, snr_ref_db=55)
snr_mid  = snr_vs_range(R_sweep, snr_ref_db=65)
snr_high = snr_vs_range(R_sweep, snr_ref_db=75)

R_low  = max_detection_range(snr_low,  R_sweep)
R_mid  = max_detection_range(snr_mid,  R_sweep)
R_high = max_detection_range(snr_high, R_sweep)

fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5.5))
fig8.suptitle('Plot 8 — Monostatic Radar Equation: SNR vs Range\n'
              'SNR drops as 1/R^4  →  -40 dB per decade of range',
              fontweight='bold')

axes8[0].plot(R_sweep, snr_low,  'g-',  lw=2, label=f'Low power (ref 55 dB)  max {R_low:.0f} m')
axes8[0].plot(R_sweep, snr_mid,  'b-',  lw=2, label=f'Medium power (ref 65 dB)  max {R_mid:.0f} m')
axes8[0].plot(R_sweep, snr_high, 'r-',  lw=2, label=f'High power (ref 75 dB)  max {R_high:.0f} m')
axes8[0].axhline(SNR_THRESHOLD_DB, color='black', ls='--', lw=1.8,
                 label=f'Detection threshold ({SNR_THRESHOLD_DB} dB)')
for R_v, col in [(R_low, 'green'), (R_mid, 'blue'), (R_high, 'red')]:
    axes8[0].axvline(R_v, color=col, ls=':', lw=1, alpha=0.6)
axes8[0].set_xlabel('Range (m)')
axes8[0].set_ylabel('SNR (dB)')
axes8[0].set_title('SNR vs Range for different transmit power levels')
axes8[0].legend(fontsize=8.5)
axes8[0].set_xlim(10, 800)
axes8[0].set_ylim(-30, 120)

# Log-log to verify 40 dB/decade slope
axes8[1].loglog(R_sweep, 10 ** (snr_mid / 10), 'b-', lw=2, label='SNR (linear scale)')
axes8[1].axhline(10 ** (SNR_THRESHOLD_DB / 10), color='black', ls='--', lw=1.5,
                 label=f'Threshold ({SNR_THRESHOLD_DB} dB)')
axes8[1].axvline(R_mid, color='blue', ls=':', lw=1.2, label=f'Max range: {R_mid:.0f} m')
decade_R = [50, 100, 200, 400]
for i in range(len(decade_R) - 1):
    axes8[1].annotate('', xy=(decade_R[i+1], 1e3),
                      xytext=(decade_R[i], 1e3),
                      arrowprops=dict(arrowstyle='<->', color='gray'))
    axes8[1].text((decade_R[i] + decade_R[i+1]) / 2, 1.5e3,
                  '−40 dB', ha='center', fontsize=8, color='gray')
axes8[1].set_xlabel('Range (m)')
axes8[1].set_ylabel('SNR (linear)')
axes8[1].set_title('Log-log: confirms 1/R^4 law (slope = −4)')
axes8[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot8_snr_vs_range.png', dpi=150, bbox_inches='tight')
print("  [OK] Saved plot8_snr_vs_range.png")


# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================
print()
print(DIVIDER)
print("  PHASE 2 RESULTS SUMMARY — MONOSTATIC BASELINE")
print(DIVIDER)
print(f"  {'Parameter':<35} {'Value':>15}")
print("-" * 60)
rows = [
    ("Range resolution",              f"{range_res:.2f} m"),
    ("Max unambiguous range",         f"{max_range:.1f} m"),
    ("Max detectable range (65 dB)",  f"{R_mid:.0f} m"),
    ("Velocity resolution",           f"{vel_res:.3f} m/s"),
    ("Max detectable velocity",       f"{max_vel:.2f} m/s"),
    ("Targets detected (Plot 4)",     f"{sum(1 for R in targets_R if any(np.abs(r_ax4[det4] - R) < 30))}/{len(targets_R)}"),
]
for name, val in rows:
    print(f"  {name:<35} {val:>15}")
print(DIVIDER)
print()
print("  Figures saved:")
for i, name in enumerate([
    "plot1_beat_signal.png       — Raw beat signal (time domain)",
    "plot2_spectrogram.png       — Chirp frequency spectrogram",
    "plot3_range_fft_snr.png    — Range FFT vs SNR levels",
    "plot4_multi_target_cfar.png — Multi-target detection + CFAR table",
    "plot5_range_doppler_map.png — 2D Range-Doppler heatmap",
    "plot6_velocity_profile.png  — Velocity extraction per target",
    "plot7_cfar_analysis.png     — CFAR threshold sweep + ROC curve",
    "plot8_snr_vs_range.png      — SNR vs range (radar equation)",
], 1):
    print(f"  {i}. {name}")
print()
print("  Done. Run plt.show() or open PNG files to view results.")
print(DIVIDER)

plt.show()
