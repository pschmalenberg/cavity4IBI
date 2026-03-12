from .mirise import MIRISEDataset
from .cavity import CavityDataset
from .steth import StethDataset

from torch.utils.data import ConcatDataset
import os 
import utils as u


paths = {
    "pardo": "",
    "mirise": "",
    #C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\dataset
    "cavity_data": u.dir_dataset
}


def get_dataset(name, split, sr):
    if name == "mirise":
        return MIRISEDataset(paths[name], split, sr)
    elif name == "cavity_data":
        return CavityDataset(paths[name], split, sr)
    elif name in ["steth_exp", "pardo", "breathing"]:
        return StethDataset(paths[name], split, sr)
    elif name == "steth_all":
        return ConcatDataset(
            [
                StethDataset(paths["steth_exp"], split, sr),
                StethDataset(paths["pardo"], split, sr),
                StethDataset(paths["breathing"], split, sr),
            ]
        )
    elif name == "all":
        return ConcatDataset(
            [
                MIRISEDataset(paths["mirise"], split, sr),
                CavityDataset(paths["cavity_data"], split, sr),
                StethDataset(paths["steth_exp"], split, sr),
                StethDataset(paths["pardo"], split, sr),
                StethDataset(paths["breathing"], split, sr),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
