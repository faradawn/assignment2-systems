# Build your own Transformer with FlashInfer

Find a Pytorch Docker Image. E.g. Vast.ai: https://hub.docker.com/r/vastai/pytorch/. 

Rent a A100 GPU.

### Install FlashInfer

```
conda create -n flashinfer python=3.10 -y
conda activate flashinfer

pip install flashinfer-python

pip install einx jaxtyping

python -m cs336_systems.benchmarking_script --max_seq_len 512 --batch_size 64
```