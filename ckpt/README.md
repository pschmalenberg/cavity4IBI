# Released checkpoint

`2025-11-22_23h02min.pth` is the final local paper checkpoint. It is a plain
PyTorch state dictionary, not a serialized model object. Instantiate the
included `models.ConvTasNet` and then call `load_state_dict`.

Identity and architecture counts are machine-readable in `manifest.csv` and
checked by `python verify_release.py`.