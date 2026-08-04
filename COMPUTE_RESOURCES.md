# Computational resources

## Reported environment

The Supplementary Information reports the following model development and
training environment:

- Windows 10 Pro
- Python 3.11.4
- PyTorch 2.0.1 and Torchaudio 2.0.2
- CUDA 11.7 and cuDNN 8.5.0
- Three NVIDIA GeForce RTX 3090 TURBO GPUs with 24 GB VRAM each
- AMD EPYC 7282 16-core processor
- 255 GB system memory

The training description specifies AdamW, batch size 8, 50 epochs, and
checkpointing every two hours. Package requirements are recorded in
[requirements.txt](requirements.txt).

## Values not reported

The available manuscript, Supplementary Information, checkpoint, and logs do
not establish:

- total wall-clock training or evaluation time;
- the number of complete training or tuning runs;
- whether or how the three GPUs were used in parallel;
- average or peak GPU utilization;
- inference latency or throughput;
- cloud or financial cost;
- electrical energy use; or
- carbon emissions.

Checkpoint timestamps and the two-hour checkpoint interval are not sufficient
to derive these values. They require author records or a new measured run and
must not be estimated as reported results.

## Practical reproduction boundary

The integrity check and retained P12 metric check are lightweight. Full
preprocessing, all-participant inference, DTW, and retraining have materially
different compute requirements and were not rerun while assembling this
release. See [README.md](README.md) for the available commands and
[REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) before retraining.