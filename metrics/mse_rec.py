import numpy as np, os
from loaders import get_dataset

def test_on_set(data_name, test_name, model_name):
    dataset = get_dataset(test_name, "test", 2000)
    y_true = np.stack([ecg for _, ecg in dataset])
    y_pred = np.load(f"results/{data_name}_on_{test_name}_{model_name}_test.npy")

    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    print(f"{data_name} tested on {test_name} ({len(dataset)} samples):")
    print(f"MSE = {mse.mean():.4f} ± {mse.std():.4f}")

    return y_true, y_pred

def evaluate_rec(data_name, test_file, model_name):
    path_true = f"results/{data_name}_on_{test_file}_{model_name}_true.npy"
    path_pred = f"results/{data_name}_on_{test_file}_{model_name}_pred.npy"
    assert os.path.exists(path_pred), "You must run predict.py first."
    
    y_true, y_pred = np.load(path_true), np.load(path_pred)
    mse = (y_true - y_pred) ** 2
    print(f"{data_name} tested on {test_file}:")
    print(f"MSE = {mse.mean():.4f} ± {mse.std():.4f}")

    return y_true, y_pred

def evaluate_rec_v2(dir_root, data_name):
    import os 
    path_true = os.path.join(dir_root, data_name+"_true.npy")
    path_pred = os.path.join(dir_root, data_name+"_pred.npy")
    assert os.path.exists(path_pred), "You must run predict.py first."
    
    y_true, y_pred = np.load(path_true), np.load(path_pred)
    mse = (y_true - y_pred) ** 2
    print(f"{data_name} tested on {data_name}:")
    print(f"MSE = {mse.mean():.4f} ± {mse.std():.4f}")

    return y_true, y_pred


if __name__ == "__main__":
    y_true, y_pred = evaluate_rec("all", "HRI_Test_09", "conv-tasnet")