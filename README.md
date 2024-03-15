# MAZero (ICLR 2024)
Open-source codebase for MAZero, from ["Efficient Multi-agent Reinforcement Learning by Planning"](https://openreview.net/forum?id=CpnKq3UJwp) at ICLR 2024.

## Environments
MAZero requires python3 (>=3.8) and pytorch (>=1.12) with the development headers. 
```
conda create -n mazero python=3.8
conda activate mazero
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install ray tensorboard matplotlib gymnasium wandb seaborn scipy opencv-python==4.5.1.48 cython==0.29.23
```

### SMAC Installation
For SMAC installation, please refer to [https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac).

## Usage

Before starting training, you need to build the c++/cython style external packages. (GCC version 7.5+ is required.)
```
cd core/mcts/ctree
bash make.sh
```

### Quick start
```
conda activate mazero
python test/test_train_sync_parallel.py
```

### Bash file
We provide `train_smac.sh` for training.
- With 1 GPUs (A100 80G): `bash train_smac.sh`


## Citation
If you find this repo useful, please cite our paper:
```
@inproceedings{liu2023efficient,
  title={Efficient Multi-agent Reinforcement Learning by Planning},
  author={Liu, Qihan and Ye, Jianing and Ma, Xiaoteng and Yang, Jun and Liang, Bin and Zhang, Chongjie},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

## Acknowledgement
We appreciate the following github repos a lot for their valuable code base implementations:

https://github.com/YeWR/EfficientZero

https://github.com/koulanurag/muzero-pytorch

https://github.com/werner-duvaud/muzero-general
