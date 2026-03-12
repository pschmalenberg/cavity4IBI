import numpy as np, neurokit2 as nk

def rr_intervals(pos_true, pos_pred, sr):
    rr_true = np.diff(pos_true).mean() * 1000 / sr
    rr_pred = np.diff(pos_pred).mean() * 1000 / sr
    print(f"Mean RR duration = {rr_true:.4f} msec")
    print(f"Predicted duration = {rr_pred:.4f} msec")
    print(f"Difference = {abs(rr_true - rr_pred):.4f} msec")
    print(f"Missed peaks: {abs(len(pos_true) - len(pos_pred))}")

def heart_rate(pos_true, pos_pred, sr):
    hr_true = nk.ecg_rate(pos_true, sampling_rate=sr)
    hr_pred = nk.ecg_rate(pos_pred, sampling_rate=sr)

    print(f"Mean HR = {hr_true.mean():.4f} BPM")
    print(f"Predicted HR = {hr_pred.mean():.4f} BPM")
    print(f"Difference = {abs(hr_true.mean() - hr_pred.mean()):.4f} BPM")
