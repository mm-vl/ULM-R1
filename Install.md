# ULM-R1 Install

## To quickly download data/model from huggingface
```shell
# git lfs clone https://hf-mirror.com/xxx
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## Janus-Pro & open_r1 environment [Training]

> Requirement: Cuda>=12.4

```shell
conda create -n corl python=3.10 -y 
conda activate corl 
git clone https://github.com/mm-vl/ULM-R1.git
cd ULM-R1
pip3 install -e . 


#pip3 install wandb==0.18.3
#pip3 install vllm==0.7.2

pip3 install flash-attn --no-build-isolation  --use-pep517  # 

# t2i environment
# >=trl-0.17.0
pip3 install git+https://ghfast.top/https://github.com/huggingface/trl.git


##
#pip3 install vllm==0.6.6.post1
##pip3 install -r requirements.txt
#pip3 install -e . 
#pip3 install wandb==0.18.3
```

## evaluation environment

### GenEval 
```shell
cd eval/t2i/geneval/mmdetection
pip3 install -v -e .

pip3 install open_clip_torch
pip3 install clip-benchmark
pip3 install openai==0.28
```


### DPG eval
```shell
conda create -n uni_eval python=3.10
conda activate uni_eval

pip install pip==24.0

cd eval/t2i/dpg_bench/
pip install -r requirements.txt
# https://github.com/modelscope/modelscope/blob/master/requirements/framework.txt
#pip install datasets==3.2.0
#pip install modelscope
#pip install simplejson
#pip install sortedcontainers
#pip install librosa==0.10.1

# https://github.com/facebookresearch/fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

```
### VLMEvalKit 
```shell
cd eval/mm2t
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .

pip3 install validators
pip3 install sty
pip3 install decord
pip3 install imageio
pip3 install timeout_decorator
pip3 install xlsxwriter
pip3 install openpyxl
``` 

