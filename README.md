<div align="center">
<h1> Co-Reinforcement Learning for<br>Unified Multimodal Understanding and Generation</h1>

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2505.17534) 
[![Github](https://img.shields.io/badge/ULM--R1-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/mm-vl/ULM-R1) 
[![Hugging Face Collection](https://img.shields.io/badge/CoRL_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/mm-vl/corl-67e0f23d6ecbdc3a9fb747e9) 

</div>

## Introduction

CoRL is a GRPO-based RL framework designed to simultaneously enhance the generation and understanding capabilities of ULMs within a shared policy optimization paradigm. It comprises a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. 

<p align="center">
   <img src="assets/corl_overview.jpg" alt="method overview." style="width: 85%;">
</p>

## 📢 Latest Updates

- [01/06/2025] 📌 Core code of CoRL.


## Environment 

```bash
git clone https://github.com/mm-vl/ULM-R1.git
cd ULM-R1
conda create -n corl python=3.10 -y
conda activate corl
pip install -e .
pip install flash-attn --no-build-isolation
# pip install flash-attn --no-build-isolation --use-pep517
```
> Pls refer to [install.md](Install.md) for more details.

## Training Data

- [Stage I: unified RL data](https://huggingface.co/datasets/mm-vl/x2x_rft_22k), with ~70\% concept coverage.
<p align="center">
   <img src="assets/data_sample.jpg" alt="training example for unified RL." style="width: 85%;">
</p>

- [Stage II: refined RL data for text-to-image](https://huggingface.co/datasets/mm-vl/x2x_rft_16k)
- [Stage II: refined RL data for multimodal understanding (MC-Format)](https://huggingface.co/datasets/mm-vl/mcot_r1_mcq_66k)
- [Stage II: refined RL data for multimodal understanding (OE-Format)](https://huggingface.co/datasets/mm-vl/mcot_r1_vqa_66k)


## Training Pipeline

- Unified RL 

```shell
bash scripts/corl_unified.sh
```

[//]: # (### Stage II: Refined RL )

[//]: # ()
[//]: # (- Multimodal Understanding)

[//]: # ()
[//]: # (```shell)

[//]: # ()
[//]: # (```)

[//]: # (- MC-Format)

[//]: # ()
[//]: # (- OE-Format)

[//]: # ()
[//]: # (- Text-to-Image Generation)

[//]: # ()
[//]: # (```shell)

[//]: # ()
[//]: # (```)

## Acknowledgement

[Janus-Pro](https://github.com/deepseek-ai/Janus) | [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) | [R1-V](https://github.com/Deep-Agent/R1-V)
