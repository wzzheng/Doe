<!-- ![logo](./assets/logo.png) --> 

This repository contains the implementation of Doe. 

> **Doe: Closed-Loop Autonomous Driving with Large World Model**<br>
<!-- > [Paper](https://arxiv.org/abs/2405.17429)  | [Project Page](https://wzzheng.net/GaussianFormer)  -->

## News.
- **[2024/12/13]** Model weights and evaluation code release.
<!-- - **[2024/12/13]** Paper released on [arXiv](https://arxiv.org/abs/2405.17429). -->
- **[2024/12/13]** Demo release.

## Doe
<!-- <img src="assets/framework.pdf"> -->

<!-- .pdf cant be rendered? How to get png ???? -->

Doe proposes a closed-loop and end-to-end large world model for unified perception, prediction, and planning for autonomous driving.

We use free-form texts (i.e., scene descriptions) for perception and generate future predictions directly in the RGB space with image tokens. For planning, we employ a position-aware tokenizer to effectively encode action into discrete tokens. We train a multi-modal transformer to autoregressively generate perception, prediction, and planning tokens in an end-to-end and unified manner. 

## Getting Started

### Data Preparation
1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

2. Download the annotations data_nusc from [OmniDrive](https://github.com/NVlabs/OmniDrive/releases/tag/v1.0) and unzip it.

3. Download the VQVAE weights from [HERE](https://github.com/facebookresearch/chameleon) and put them to the following directory as [HERE](https://github.com/Alpha-VLLM/Lumina-mGPT):

```
Doe/
- model/
    - lumina_mgpt/
        - ckpts/
            - chameleon/
                - tokenizer/
                    - text_tokenizer.json
                    - vqgan.yaml
                    - vqgan.ckpt
    - xllmx/
- ...
```

### Inference

<!-- We provide the following checkpoints: -->

1. Generate the conversation data for inference and set the max :
```bash
# max length: 1 for qa, 5 for planning
python dataset/gen_data.py \
--info_path path/to/infos_var.pkl \
--qa_path path/to/OmniDriveDataset \
--nusc_path path/to/nuscenes \
--save_path path/to/save/outputs \
--max_length 1
```

2. Inference with a model ckpt:
```bash
# set split and id for multi gpus
CUDA_VISIBLE_DIVICES=0 python inference/eval.py \
--anno_path path/to/val_infos.pkl \
--nusc_path path/to/nuscenes \
--save_path path/to/save/output \
--model_path path/to/model/ckpt \
--data_path path/to/generated/data.json \
--task qa
```

## Related Projects

Our code is originally based on the excellent work [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT).

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{huang2024gaussian,
    title={GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction},
    author={Huang, Yuanhui and Zheng, Wenzhao and Zhang, Yunpeng and Zhou, Jie and Lu, Jiwen},
    journal={arXiv preprint arXiv:2405.17429},
    year={2024}
}
@article{huang2024probabilisticgaussiansuperpositionefficient,
      title={GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction}, 
      author={Yuanhui Huang and Amonnut Thammatadatrakoon and Wenzhao Zheng and Yunpeng Zhang and Dalong Du and Jiwen Lu},
      journal={arXiv preprint arXiv:2412.04384},
      year={2024}
}
```
