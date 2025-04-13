<h2 align="center">TinyLLaVA-Video-R1</a><h5 align="center">

<div align="center">

[Xingjian Zhang](https://scholar.google.com/citations?user=H34fwioAAAAJ&hl=zh-CN)<sup>1*</sup>,
[Siwei Wen](https://scholar.google.com/citations?user=kJRiUYwAAAAJ&hl=zh-CN)<sup>1,2*</sup>,
[Wenjun Wu](https://iai.buaa.edu.cn/info/1013/1093.htm)<sup>1,2</sup>, 
[Lei Huang](https://huangleibuaa.github.io/)<sup>1,2,✉</sup>

<sup>1</sup>SKLCCSE, Institute of Artificial Intelligence, Beihang University, <br>
<sup>2</sup>Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Beihang University

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.14905-AD1C18.svg?logo=arXiv)](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)
[![GitHub issues](https://img.shields.io/github/issues/ZhangXJ199/TinyLLaVA-Video-R1?color=critical&label=Issues)](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)
[![GitHub Stars](https://img.shields.io/github/stars/ZhangXJ199/TinyLLaVA-Video-R1?style=social)](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)

</div>

## 📰 News

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> About
TinyLLaVA-Video-R1 is a small-scale video reasoning model built upon the fully open-source [TinyLLaVA-Video](https://github.com/ZhangXJ199/TinyLLaVA-Video) framework. Designed for researchers with limited computational resources, it leverages reinforcement learning to enhance reasoning abilities while maintaining a model size under 4 billion parameters. TinyLLaVA-Video-R1 demonstrates improved video question-answering performance and reflective reasoning behaviors ("aha moments"). The model and training process are fully traceable, ensuring reproducibility and reliability. This repository provides the model, code, and experimental setups for easy replication

## 🛠️ Installation

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/ZhangXJ199/TinyLLaVA-Video-R1.git
cd TinyLLaVA-Video-R1
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n tinyllava_video python=3.10 -y
conda activate tinyllava_video
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages
```Shell
pip install flash-attn --no-build-isolation
```
##### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## 📌 Usage

### 1. Data Preparation


### 2. Train


### 3. Evaluation

We currently provide evaluations on 4 benchmarks, including [Video-MME](https://video-mme.github.io/home_page.html#leaderboard), [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [MLVU](https://github.com/JUNJIE99/MLVU), [MMVU](https://github.com/yale-nlp/MMVU).

#### Video-MME

1. Download [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) and put it under ``path/to/your/dataset/eval/Video-MME``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR``, ``conv-mode`` and ``duration`` in ``scripts/eval/videomme.sh``. There are three types of ``duration`` available for testing: ``short``, ``medium``, and ``long``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/videomme.sh
   ```

#### MVBench

1. Download [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) and put it under ``path/to/your/dataset/eval/MVBench``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR`` and ``conv-mode`` in ``scripts/eval/mvbench.sh``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mvbench.sh
   ```

#### MLVU

1. Download [MLVU](https://huggingface.co/datasets/MLVU/MVLU) and put it under ``path/to/your/dataset/eval/MLVU``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR`` and ``conv-mode`` in ``scripts/eval/mlvu.sh``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mlvu.sh
   ```

#### MMVU

1. Download [MMVU](https://huggingface.co/datasets/yale-nlp/MMVU) and put it under ``path/to/your/dataset/eval/MMVU``.
2. Please change ``MODEL_PATH``, ``MODEL_NAME``, ``EVAL_DIR`` and ``conv-mode`` in ``scripts/eval/mmvu.sh``.
3. Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvu.sh

## 📊 Results

## 😄 Acknowledgement

## 📨 Contact

## 📝 Citation
