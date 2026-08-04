from .cavity import CavityDataset

import utils as u


def get_dataset(name, split, sr):
    if name == "cavity_data":
        return CavityDataset(u.dir_dataset, split, sr)
    raise ValueError(f"Unknown dataset: {name}")


__all__ = ["CavityDataset", "get_dataset"]