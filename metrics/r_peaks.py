import neurokit2 as nk, numpy as np


def r_peak_accuracy(y_true, y_pred, sr, error=0.005, return_pos=False):
    correct, total = 0, 0
    for i in range(len(y_true)):
        ecg_true = nk.ecg_clean(y_true[i], sampling_rate=sr)
        ecg_pred = nk.ecg_clean(y_pred[i].squeeze(), sampling_rate=sr)

        pos_true = nk.ecg_peaks(ecg_true, sampling_rate=2000)[1]["ECG_R_Peaks"]
        pos_pred = nk.ecg_peaks(ecg_pred, sampling_rate=2000)[1]["ECG_R_Peaks"]

        diff = np.zeros(len(pos_true))
        for i, pos in enumerate(pos_true):
            closest_i = np.argmin(abs(pos - pos_pred))
            diff[i] = abs(pos - pos_pred[closest_i])
        
        within = np.where(diff <= error * sr)[0]
        correct += len(within)
        total += len(diff)

    print(f"Correct: {correct}, Total: {total}")
    print("Accuracy: {:.3f}".format(correct / total))

    assert not return_pos or len(y_true) == 1, "Will return positions for whole files"
    if return_pos:
        return pos_true, pos_pred
