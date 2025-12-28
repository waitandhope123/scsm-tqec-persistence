#!/usr/bin/env python3
# toy15_final_FIXED.py - NaN-safe version
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from scipy import signal

class Params:
    def __init__(self):
        self.seed = 1234
        self.n_samples = 5000
        self.fs = 5000.0
        self.snr_db_range = [-5, 0, 5]
        self.drift_rate = 0.005
        self.t = np.arange(self.n_samples) / self.fs
        self.dt = 1.0 / self.fs
        self.drive_freqs_hz = np.logspace(2.5, 3.5, 8)

params = Params()
np.random.seed(params.seed)

def generate_threshold_chi(t, drive, p):
    omega_0 = 2000 * 2 * np.pi
    damping = 0.1 * omega_0
    threshold = 0.3
    x = np.zeros_like(t)
    dxdt = np.zeros_like(t)
    
    for i in range(1, len(t)):
        nonlinear = 3 * (x[i-1]**3 if abs(x[i-1]) > threshold else 0)
        d2xdt2 = -damping * dxdt[i-1] - omega_0**2 * x[i-1] - nonlinear + drive[i-1]
        dxdt[i] = dxdt[i-1] + d2xdt2 * p.dt
        x[i] = x[i-1] + dxdt[i] * p.dt
    return x

def generate_colored_noise(t, drive, p):
    white = np.random.randn(len(t))
    return signal.lfilter([1], [1, -0.99], white) * 0.1

truth_models = {"threshold_chi": generate_threshold_chi}
null_models = {"colored_noise": generate_colored_noise}

def generate_drive(p, freq_hz):
    return signal.chirp(p.t, f0=0.8*freq_hz, f1=1.2*freq_hz, t1=p.n_samples*p.dt)

def generate_dataset(model_fn, model_name, p, freq_idx):
    drive = generate_drive(p, p.drive_freqs_hz[freq_idx])
    clean = model_fn(p.t, drive, p)
    snr_db = p.snr_db_range[freq_idx % len(p.snr_db_range)]
    noise_scale = 10**(-snr_db / 20.0) * np.std(clean)
    noise = noise_scale * np.random.randn(len(p.t))
    drift = p.drift_rate * np.cumsum(np.random.randn(len(p.t)))
    return {"t": p.t, "drive": drive, "response": clean + noise + drift, "snr_db": snr_db, "freq_hz": p.drive_freqs_hz[freq_idx]}

def extract_fingerprints(data):
    drive, response = data["drive"], data["response"]
    
    # Coherence (safe)
    f, coh = signal.coherence(drive, response, fs=params.fs, nperseg=512)
    coh_peak = float(np.nanmax(coh))
    
    # Phase (NaN-safe)
    hilb_drive = signal.hilbert(drive)
    hilb_resp = signal.hilbert(response)
    phase_ratio = hilb_resp / hilb_drive
    phase_valid = ~np.isnan(phase_ratio) & ~np.isinf(phase_ratio)
    phase_lag = np.unwrap(np.angle(phase_ratio[phase_valid])) if np.any(phase_valid) else np.array([])
    phase_std = float(np.nanstd(phase_lag)) if len(phase_lag) > 1 else 0.0
    
    # Harmonics (NaN-safe)
    f_psd, psd = signal.welch(response, fs=params.fs, nperseg=512)
    drive_freq = data["freq_hz"]
    fund_idx = np.argmin(np.abs(f_psd - drive_freq))
    harm_idx = np.argmin(np.abs(f_psd - 2*drive_freq))
    harm_ratio = float(psd[harm_idx] / psd[fund_idx]) if harm_idx < len(psd) and psd[fund_idx] > 0 else 0.0
    
    return {
        "coherence_peak": coh_peak,
        "phase_std": phase_std,
        "harmonic_ratio": harm_ratio,
        "snr_db": data["snr_db"]
    }

def score_model(data):
    fp = extract_fingerprints(data)
    is_match = (fp["harmonic_ratio"] > 0.05 and fp["phase_std"] < 30.0)
    score = np.mean([fp["coherence_peak"], 1.0 / (1 + fp["phase_std"] / 10)])
    return {"fingerprints": fp, "is_recovered": is_match, "score": float(score)}

PASS_CRITERIA = {"recovery_rate": ">50% truth, <30% null FP"}

def run_toy15():
    output_dir = Path("toy15_results")
    output_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir()
    
    print("🧪 Toy15: Testing χ nonlinearity detectability...")
    
    truth_results, null_results = [], []
    
    # 5 truth trials
    for i in range(5):
        data = generate_dataset(truth_models["threshold_chi"], "truth", params, i)
        score = score_model(data)
        truth_results.append(score)
        print(f"Truth {i}: harm={score['fingerprints']['harmonic_ratio']:.3f}, recovered={score['is_recovered']}")
    
    # 5 null trials
    for i in range(5):
        data = generate_dataset(null_models["colored_noise"], "null", params, i)
        score = score_model(data)
        null_results.append(score)
        print(f"Null  {i}: harm={score['fingerprints']['harmonic_ratio']:.3f}, recovered={score['is_recovered']}")
    
    truth_recovery = np.mean([r["is_recovered"] for r in truth_results])
    null_fp = np.mean([r["is_recovered"] for r in null_results])
    
    status = "PASS" if (truth_recovery > 0.5 and null_fp < 0.3) else "FAIL"
    
    summary = {
        "status": status,
        "truth_recovery": float(truth_recovery),
        "null_fp": float(null_fp),
        "interpretation": f"χ survives active probing: {truth_recovery:.0%} truth vs {null_fp:.0%} null FP"
    }
    
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    # NaN-SAFE PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Filter NaNs for histogram
    truth_harm = np.array([r["fingerprints"]["harmonic_ratio"] for r in truth_results])
    null_harm = np.array([r["fingerprints"]["harmonic_ratio"] for r in null_results])
    
    truth_harm_clean = truth_harm[~np.isnan(truth_harm)]
    null_harm_clean = null_harm[~np.isnan(null_harm)]
    
    if len(truth_harm_clean) > 0:
        ax1.hist(truth_harm_clean, bins=5, alpha=0.7, label="χ Truth", color='green')
    if len(null_harm_clean) > 0:
        ax1.hist(null_harm_clean, bins=5, alpha=0.7, label="Null", color='red')
    ax1.axvline(0.05, color='black', ls='--', label="Threshold")
    ax1.legend()
    ax1.set_xlabel("Harmonic Ratio")
    ax1.set_title("Nonlinearity Fingerprint")
    
    ax2.bar(["Truth Recov.", "Null FP"], [truth_recovery, null_fp], color=['green', 'red'])
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color='black', ls='--')
    ax2.set_ylabel("Rate")
    
    plt.tight_layout()
    plt.savefig(run_dir / "results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎯 FINAL VERDICT: {status}")
    print(f"Truth: {truth_recovery:.0%} | Null FP: {null_fp:.0%}")
    print(f"SCSM χ: {'DETECTABLE' if status=='PASS' else 'REQUIRES STRONGER FINGERPRINTS'}")
    print(f"Saved: {run_dir}")

if __name__ == "__main__":
    run_toy15()
