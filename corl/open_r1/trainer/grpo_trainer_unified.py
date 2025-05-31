# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.nn as nn
import torch.utils.data
import transformers
from packaging import version
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import gather, is_peft_model, set_seed

from trl.import_utils import is_deepspeed_available
from trl import ScriptArguments
from trl.trainer.callbacks import SyncRefModelCallback
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import selective_log_softmax

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from janus.models import VLChatProcessor, MultiModalityCausalLM

# What we call a reward function is a callable that takes
# a list of prompts and completions and returns a list of rewards.
# When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


class JanusProUnifiedGRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,  # inherit from TrainingArguments

            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[
                Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
            ] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[
                Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
            ] = (None, None),
            attn_implementation: str = "flash_attention_2",
            peft_config: Optional["PeftConfig"] = None,
            task_args: ScriptArguments = None,
    ):
        self.task_args = task_args
        self.task_format = task_args.task_format

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # ******************* Models *******************
        # 1. Trained policy model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto"
                    or torch_dtype is None):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a"
                    f"string representing a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Janus" in model_id:
                model = AutoModelForCausalLM.from_pretrained(
                    model, trust_remote_code=True, torch_dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        # **************************************
        model = self.init_trainable_parameters(model)

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`")
            model = get_peft_model(model, peft_config)

        # 2. Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            if "Janus" in model_id:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs)

        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter
            # can be disabled to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT config is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # 3. Processing class
        if processing_class is None:
            if "Janus" in model_id:
                processing_class = VLChatProcessor.from_pretrained(model_id)
                # processing_class.system_prompt = SYSTEM_PROMPT
                processing_class.system_prompt = ""
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )

        # ******************* Reward functions *******************
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        # self.reward_funcs = reward_funcs
        self.t2i_reward_funcs = [rf for rf in reward_funcs if "t2i" in rf.__name__]
        self.mm2t_reward_funcs = [rf for rf in reward_funcs if "t2i" not in rf.__name__]

        # 1. Reward weights
        if args.reward_weights is not None:  # list[float]
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number "
                    f"of reward functions ({len(reward_funcs)})"
                )
            # self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
            self.t2i_reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
            self.mm2t_reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            # self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
            self.t2i_reward_weights = torch.ones(len(self.t2i_reward_funcs), dtype=torch.float32)
            self.mm2t_reward_weights = torch.ones(len(self.mm2t_reward_funcs), dtype=torch.float32)
        # 2. Reward
        self.init_external_model_for_rewards()

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of
        # elements in the input tensor associated with the key "input_ids". However, in GRPO,
        # the sampled data does not include the "input_ids" key. Instead, the available keys is
        # "prompt". As a result, the trainer issues the warning: "Could not estimate the number of
        # tokens of the input, floating-point operations will not be computed." To suppress this
        # warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to
        # True. This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        # self._total_train_tokens = 0
        # self.log_completions = args.log_completions
        # self.num_completions_to_print = args.num_completions_to_print

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # ******************* Training arguments *******************
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon  # defalt=0.2
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        self.max_prompt_length = args.max_prompt_length  # prompt+image
        self.max_completion_length = args.max_completion_length  # = |o_i| in GRPO
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # param for .generate()
        self.temperature = args.temperature

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            # temperature=self.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.tokenizer.eos_token_id,
            bos_token_id=processing_class.tokenizer.bos_token_id,
            eos_token_id=processing_class.tokenizer.eos_token_id,
            #
            # top_p=args.top_p,
            # top_k=args.top_k,
            # min_p=args.min_p,
            # repetition_penalty=args.repetition_penalty,
        )
        self.t2i_generation_kwargs = {
            "cfg_weight": 5,
            "parallel_size": self.num_generations,
            "temperature": 1,  # HACK
            # "temperature": self.temperature,
            "image_token_num_per_image": 576,  # 24x24
            "img_size": 384,
            "patch_size": 16,
            "pad_id": processing_class.pad_id,  # !!
            "seed": self.args.seed,
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when
        # generating with transformers if num_generations exceeds per_device_train_batch_size.
        # We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class
        # depends on whether the model accepts loss-related kwargs.
        # Since we compute our own loss, this check is irrelevant.
        # We set self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(
                ref_model=self.ref_model, accelerator=self.accelerator))

    @staticmethod
    def init_trainable_parameters(model):
        # model.language_model.config._attn_implementation == attn_implementation

        # fix gen_vision_model and vision_model
        for param in model.gen_vision_model.parameters():
            param.requires_grad = False
        for param in model.vision_model.parameters():
            param.requires_grad = False

        # fix gen_aligner and aligner
        for param in model.gen_aligner.parameters():
            param.requires_grad = False
        for param in model.aligner.parameters():
            param.requires_grad = False

        # fix gen_head and gen_embed ???
        for param in model.gen_head.parameters():
            param.requires_grad = False
        for param in model.gen_embed.parameters():
            param.requires_grad = False

        # trainable: llm, (gen_embed, gen_head)

        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable()

        return model

    def init_external_model_for_rewards(self):
        if "t2i_clip_reward" in self.task_args.reward_funcs:
            print("using reward_func: t2i_clip_reward")
            # Load reward model
            from corl.open_r1.rewards.open_clip import create_model_and_transforms, get_tokenizer
            clip_model, _, clip_preprocess = create_model_and_transforms(
                'ViT-H-14',
                f'{self.task_args.model_ckpt_dir}/open_clip_pytorch_model.bin',
                precision='amp',
                device="cuda",
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            clip_tokenizer = get_tokenizer('ViT-H-14')
            clip_model = clip_model.to("cuda")
            clip_model.eval()

            self.clip_preprocess = clip_preprocess
            self.clip_tokenizer = clip_tokenizer
            self.clip_model = clip_model

        if ("t2i_cycle_reward" in self.task_args.reward_funcs or
                "t2i_bid_cycle_reward" in self.task_args.reward_funcs
        ):
            print("using reward_func: cycle_reward")
            from transformers import BlipProcessor, BlipForConditionalGeneration

            blip_processor = BlipProcessor.from_pretrained(
                f"{self.task_args.model_ckpt_dir}/{self.task_args.blip_model_ckpt}")
            blip_model = BlipForConditionalGeneration.from_pretrained(
                f"{self.task_args.model_ckpt_dir}/{self.task_args.blip_model_ckpt}")

            blip_model = blip_model.to("cuda")
            blip_model.eval()

            self.blip_processor = blip_processor
            self.blip_model = blip_model

            from torchmetrics.text.bert import BERTScore
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            scorer = BERTScore(
                model_name_or_path=f"{self.task_args.model_ckpt_dir}/bert-base-uncased",
                device="cuda",
            )
            lpip_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg', normalize=True).to("cuda")

            self.bert_scorer = scorer
            self.lpip_metric = lpip_metric

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method,
        # hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and reference model
    def _get_per_token_logps(
            self, model,
            mm2t_input_ids=None, mm2t_images_seq_mask=None,
            mm2t_pixel_values=None, mm2t_images_emb_mask=None,
            mm2t_logits_to_keep=None, mm2t_attention_mask=None,
            t2i_inputs_ids=None, t2i_attention_mask=None,
            t2i_discrete_img_ids=None, t2i_logits_to_keep=None,
            return_hidden_states=False,
    ):
        t2i_discrete_img_ids = t2i_discrete_img_ids.to(t2i_inputs_ids.dtype)

        mm2t_logits, t2i_logits = model(
            mm2t_input_ids=mm2t_input_ids,
            mm2t_images_seq_mask=mm2t_images_seq_mask,
            mm2t_pixel_values=mm2t_pixel_values,
            mm2t_images_emb_mask=mm2t_images_emb_mask,
            mm2t_attention_mask=mm2t_attention_mask,
            mm2t_logits_to_keep=mm2t_logits_to_keep + 1,

            t2i_input_ids=t2i_inputs_ids,
            t2i_attention_mask=t2i_attention_mask,
            t2i_discrete_img_ids=t2i_discrete_img_ids,
            t2i_logits_to_keep=t2i_logits_to_keep,

            task='unify'
        )
        # mm2t_logits: exclude the last logit: it corresponds to the next token pred
        mm2t_logits = mm2t_logits[:, :-1, :]  # (B, L-1, V)

        # exclude the first input ID since we don't have logits for it
        mm2t_input_ids = mm2t_input_ids[:, -mm2t_logits_to_keep:]  # (B, L-1)
        # For transformers<=4.48, logits_to_keep argument isn't supported
        # logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        mm2t_logits = mm2t_logits / self.temperature
        t2i_logits = t2i_logits / self.temperature

        # Compute the log probabilities for the input tokens.
        return selective_log_softmax(mm2t_logits, mm2t_input_ids), selective_log_softmax(
            t2i_logits, t2i_discrete_img_ids)

    def wrap_mm2t_prompt(self, inputs, device):
        prompts = [x["qa_prompt"] for x in inputs]
        images = [[x["real_image"]] for x in inputs]  # PIL.Image

        prompt_inputs = self.processing_class(
            conversations=prompts, images=images, force_batchify=True,
        ).to(device)

        return prompt_inputs, prompts

    def wrap_t2i_prompt(self, inputs, device=None):
        prompts = []
        for xp in inputs:
            sft_format = self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=xp["prompt"],
                sft_format=self.processing_class.sft_format,
                system_prompt="",
            )
            prompt = sft_format + self.processing_class.image_start_tag
            prompts.append(prompt)

        # self.processing_class.tokenizer.pad_token_id = self.processing_class.pad_id = 100002
        prompt_inputs = self.processing_class.tokenizer(
            prompts,
            padding=True,
            # max_length=self.max_prompt_length,
            # truncation=True,
            padding_side="left",  # default
            return_tensors="pt",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        return prompt_inputs, prompts

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device

        # Prepare inputs for text-to-image generation
        t2i_prompt_inputs, t2i_prompts = self.wrap_t2i_prompt(inputs, device)
        t2i_prompt_ids = t2i_prompt_inputs["input_ids"]
        t2i_prompt_mask = t2i_prompt_inputs["attention_mask"]
        t2i_prompts = [
            pp.strip('<|User|>:').strip('<|Assistant|>:<begin_of_image>').strip()
            for pp in t2i_prompts
        ]
        if self.max_prompt_length is not None:
            t2i_prompt_ids = t2i_prompt_ids[:, -self.max_prompt_length:]
            t2i_prompt_mask = t2i_prompt_mask[:, -self.max_prompt_length:]

        # Prepare inputs for multimodal understanding
        mm2t_prompt_inputs, mm2t_prompts = self.wrap_mm2t_prompt(inputs, device)
        mm2t_prompt_ids = mm2t_prompt_inputs["input_ids"]  # [bs, n_prompt+n_img]
        mm2t_prompt_mask = mm2t_prompt_inputs["attention_mask"]  # [bs, n_prompt+n_img]
        mm2t_images_in_prompt_mask = mm2t_prompt_inputs["images_seq_mask"]  # [bs, n_prompt+n_img]
        mm2t_pixel_values = mm2t_prompt_inputs["pixel_values"]  # [bs, 1, 3, 384, 384]
        mm2t_images_emb_mask = mm2t_prompt_inputs["images_emb_mask"]  # [bs, 1, 576]
        # for prompt, soft_prompt in zip(mm2t_prompts, mm2t_prompt_inputs["sft_format"]):
        #     if prompt[0]['role'] == "<|User|>":
        #         prompt[0]['content'] = soft_prompt
        if self.max_prompt_length is not None:
            mm2t_prompt_ids = mm2t_prompt_ids[:, -self.max_prompt_length:]
            mm2t_prompt_mask = mm2t_prompt_mask[:, -self.max_prompt_length:]
            mm2t_images_in_prompt_mask = mm2t_images_in_prompt_mask[:, -self.max_prompt_length:]

        # ******************************************************************************
        # === Generate completion ===
        torch.set_grad_enabled(False)
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            unwrapped_model.language_model.config.use_cache = False
            unwrapped_model.language_model.gradient_checkpointing_disable()

            # generate images: discrete image IDs,
            t2i_completion_ids, t2i_completions = unwrapped_model.t2i_generate_parallel(
                input_ids=t2i_prompt_ids, attention_mask=t2i_prompt_mask,
                **self.t2i_generation_kwargs
            )

            # generate texts
            inputs_embeds = unwrapped_model.prepare_inputs_embeds(
                input_ids=mm2t_prompt_ids,
                images_seq_mask=mm2t_images_in_prompt_mask,
                pixel_values=mm2t_pixel_values,
                images_emb_mask=mm2t_images_emb_mask,
            )  # [bs, n_prompt+n_img, dim]
            mm2t_completion_ids = unwrapped_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=mm2t_prompt_mask,
                generation_config=self.generation_config
            )  # [bs * num_generations, n_completion]

        t2i_prompt_ids = t2i_prompt_ids.repeat_interleave(self.num_generations, dim=0)
        t2i_prompt_mask = t2i_prompt_mask.repeat_interleave(self.num_generations, dim=0)
        t2i_completion_mask = torch.ones_like(t2i_completion_ids).int()
        # we only need to compute the logits for the completion tokens
        t2i_logits_to_keep = t2i_completion_ids.size(1)

        # Mask everything after the first EOS token
        is_eos = mm2t_completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), eos_idx)
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        mm2t_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        mm2t_prompt_ids = mm2t_prompt_ids.repeat_interleave(self.num_generations, dim=0)
        mm2t_prompt_mask = mm2t_prompt_mask.repeat_interleave(self.num_generations, dim=0)
        mm2t_images_in_prompt_mask = mm2t_images_in_prompt_mask.repeat_interleave(
            self.num_generations, dim=0)
        mm2t_pixel_values = mm2t_pixel_values.repeat_interleave(self.num_generations, dim=0)
        mm2t_images_emb_mask = mm2t_images_emb_mask.repeat_interleave(self.num_generations, dim=0)
        # Concatenate prompt_mask with completion_mask for logit computation
        mm2t_attention_mask = torch.cat([mm2t_prompt_mask, mm2t_completion_mask], dim=1)  # (B, P+C)
        mm2t_prompt_completion_ids = torch.cat([mm2t_prompt_ids, mm2t_completion_ids], dim=1)
        mm2t_completion_images_seq_mask = torch.zeros_like(
            mm2t_completion_mask).to(mm2t_images_in_prompt_mask)
        mm2t_images_seq_mask = torch.cat([
            mm2t_images_in_prompt_mask, mm2t_completion_images_seq_mask
        ], dim=1)
        # we only need to compute the logits for the completion tokens
        mm2t_logits_to_keep = mm2t_completion_ids.size(1)

        mm2t_completions = self.processing_class.tokenizer.batch_decode(
            mm2t_completion_ids, skip_special_tokens=True
        )

        # ******************************************************************************
        # === Compute rewards ===
        t2i_prompts = [pp for pp in t2i_prompts for _ in range(self.num_generations)]
        t2i_rewards_per_func = torch.zeros(
            len(t2i_prompts), len(self.t2i_reward_funcs), device=device)
        for i, (reward_func) in enumerate(self.t2i_reward_funcs):
            if "t2i_bidirectional_cycle" in reward_func.__name__:
                reward_kwargs = {
                    key: [example[key] for example in inputs for _ in range(self.num_generations)]
                    for key in inputs[0].keys() if key in ["real_image"]
                }
                # [bs * parallel_size, 3, 384, 384]
                output_reward_func = reward_func(
                    completions=t2i_completions,
                    prompts=t2i_prompts,
                    caption_model=self.blip_model,
                    caption_processor=self.blip_processor,
                    bert_scorer=self.bert_scorer,
                    lpip_metric=self.lpip_metric,
                    **reward_kwargs
                )
            elif "t2i_match_reward" in reward_func.__name__:
                output_reward_func = reward_func(
                    completions=t2i_completions, prompts=t2i_prompts,
                    mmgpt=self.model, processing_class=self.processing_class,
                )
            elif "t2i_qa" in reward_func.__name__:
                reward_kwargs = {
                    key: [example[key] for example in inputs for _ in range(self.num_generations)]
                    for key in inputs[0].keys() if key in ["qa_problem", "qa_solution"]
                }
                output_reward_func = reward_func(
                    completions=t2i_completions, prompts=t2i_prompts,
                    mmgpt=self.model, processing_class=self.processing_class,
                    gen_config=self.generation_config,
                    **reward_kwargs
                )
            elif "t2i_obj_cls" in reward_func.__name__:
                reward_kwargs = {
                    key: [example[key] for example in inputs for _ in range(self.num_generations)]
                    for key in inputs[0].keys() if key in ["cls_problem", "cls_solution"]
                }
                output_reward_func = reward_func(
                    completions=t2i_completions, prompts=t2i_prompts,
                    mmgpt=self.model, processing_class=self.processing_class,
                    gen_config=self.generation_config,
                    **reward_kwargs
                )
            elif "t2i_clip" in reward_func.__name__:
                # [bs * parallel_size, 3, 384, 384]
                output_reward_func = reward_func(
                    completions=t2i_completions,
                    prompts=t2i_prompts,
                    clip_model=self.clip_model,
                    clip_tokenizer=self.clip_tokenizer,
                    # **reward_kwargs
                )
            elif "t2i_cycle" in reward_func.__name__:
                # [bs * parallel_size, 3, 384, 384]
                output_reward_func = reward_func(
                    completions=t2i_completions,
                    prompts=t2i_prompts,
                    caption_model=self.blip_model,
                    caption_processor=self.blip_processor,
                    bert_scorer=self.bert_scorer,
                    # **reward_kwargs
                )
            else:
                raise ValueError(f"Unknown reward function: {reward_func}")

            output_reward_func = [
                reward if reward is not None else torch.nan for reward in output_reward_func
            ]  # Convert None values to NaN
            t2i_rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device)
        t2i_rewards_per_func = gather(t2i_rewards_per_func)
        # Apply weights to each reward function's output and sum
        t2i_rewards = (t2i_rewards_per_func * self.t2i_reward_weights.to(
            device).unsqueeze(0)).nansum(dim=1)

        mm2t_prompts = [pp for pp in mm2t_prompts for _ in range(self.num_generations)]
        mm2t_rewards_per_func = torch.zeros(
            len(mm2t_prompts), len(self.mm2t_reward_funcs), device=device)
        for i, (reward_func) in enumerate(self.mm2t_reward_funcs):
            if "qa" in reward_func.__name__:
                reward_kwargs = {
                    key: [example[key] for example in inputs for _ in range(self.num_generations)]
                    for key in inputs[0].keys() if key in ["qa_problem", "qa_solution", 'qa_type']
                }
                output_r_func = reward_func(completions=mm2t_completions, **reward_kwargs)
            else:  # formating
                output_r_func = reward_func(completions=mm2t_completions)

            output_r_func = [
                reward if reward is not None else torch.nan for reward in output_r_func
            ]  # Convert None values to NaN
            mm2t_rewards_per_func[:, i] = torch.tensor(output_r_func, dtype=torch.float32, device=device)
        mm2t_rewards_per_func = gather(mm2t_rewards_per_func)
        mm2t_rewards = (mm2t_rewards_per_func * self.mm2t_reward_weights.to(
            device).unsqueeze(0)).nansum(dim=1)

        # ******************************************************************************
        # Compute grouped-wise rewards ===
        if self.task_args.unify_reward:
            unified_rewards = t2i_rewards + 0.8 * mm2t_rewards

            mean_grouped_rewards = unified_rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = unified_rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)
            unified_advantages = (unified_rewards - mean_grouped_rewards) / (
                        std_grouped_rewards + 1e-4)
            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(t2i_prompts),
                (self.accelerator.process_index + 1) * len(t2i_prompts),
            )
            unified_advantages = unified_advantages[process_slice]
            t2i_advantages = None
            mm2t_advantages = None

            self._metrics["reward_unified"].append(mean_grouped_rewards.mean().item())
            # self._metrics["reward_std_unified"].append(std_grouped_rewards.mean().item())
        else:
            t2i_mean_grouped_rewards = t2i_rewards.view(-1, self.num_generations).mean(dim=1)
            t2i_std_grouped_rewards = t2i_rewards.view(-1, self.num_generations).std(dim=1)
            # Normalize the rewards to compute the advantages
            t2i_mean_grouped_rewards = t2i_mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)
            t2i_std_grouped_rewards = t2i_std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)

            t2i_advantages = (t2i_rewards - t2i_mean_grouped_rewards) / (
                    t2i_std_grouped_rewards + 1e-4)
            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(t2i_prompts),
                (self.accelerator.process_index + 1) * len(t2i_prompts),
            )
            t2i_advantages = t2i_advantages[process_slice]

            mm2t_mean_grouped_rewards = mm2t_rewards.view(-1, self.num_generations).mean(dim=1)
            mm2t_std_grouped_rewards = mm2t_rewards.view(-1, self.num_generations).std(dim=1)
            # Normalize the rewards to compute the advantages
            mm2t_mean_grouped_rewards = mm2t_mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)
            mm2t_std_grouped_rewards = mm2t_std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0)

            mm2t_advantages = (mm2t_rewards - mm2t_mean_grouped_rewards) / (
                    mm2t_std_grouped_rewards + 1e-4)
            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(mm2t_prompts),
                (self.accelerator.process_index + 1) * len(mm2t_prompts),
            )
            mm2t_advantages = mm2t_advantages[process_slice]

            unified_advantages = None

            self._metrics["reward_t2i"].append(t2i_mean_grouped_rewards.mean().item())
            # self._metrics["reward_std_t2i"].append(t2i_std_grouped_rewards.mean().item())
            self._metrics["reward_mm2t"].append(mm2t_mean_grouped_rewards.mean().item())
            # self._metrics["reward_std_mm2t"].append(mm2t_std_grouped_rewards.mean().item())

        for i, rf_name in enumerate([rf.__name__ for rf in self.t2i_reward_funcs]):
            mean_rewards = torch.nanmean(t2i_rewards_per_func[:, i]).item()
            self._metrics[f"rewards_t2i/{rf_name}/mean"].append(mean_rewards)
            # std_rewards = nanstd(t2i_rewards_per_func[:, i]).item()
            # self._metrics[f"rewards_t2i/{rf_name}/std"].append(std_rewards)

        for i, rf_name in enumerate([rf.__name__ for rf in self.mm2t_reward_funcs]):
            mean_rewards = torch.nanmean(mm2t_rewards_per_func[:, i]).item()
            self._metrics[f"rewards_mm2t/{rf_name}/mean"].append(mean_rewards)
            # std_rewards = nanstd(mm2t_rewards_per_func[:, i]).item()
            # self._metrics[f"rewards_mm2t/{rf_name}/std"].append(std_rewards)

        # ******************************************************************************
        # joint forward
        # ******************************************************************************
        # === 4. Compute logprobs ===
        torch.set_grad_enabled(True)
        with torch.inference_mode():
            if self.num_iterations > 1:
                self.model.module.language_model.gradient_checkpointing_enable()
                mm2t_old_per_token_logps, t2i_old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    mm2t_input_ids=mm2t_prompt_completion_ids,
                    mm2t_images_seq_mask=mm2t_images_seq_mask,
                    mm2t_pixel_values=mm2t_pixel_values,
                    mm2t_images_emb_mask=mm2t_images_emb_mask,
                    mm2t_attention_mask=mm2t_attention_mask,
                    mm2t_logits_to_keep=mm2t_logits_to_keep,

                    t2i_inputs_ids=t2i_prompt_ids,
                    t2i_attention_mask=t2i_prompt_mask,
                    t2i_discrete_img_ids=t2i_completion_ids,
                    t2i_logits_to_keep=t2i_logits_to_keep,
                )
            else:
                mm2t_old_per_token_logps = None
                t2i_old_per_token_logps = None

            if self.beta == 0.0:
                mm2t_ref_per_token_logps = None
                t2i_ref_per_token_logps = None
            elif self.ref_model is not None:
                self.ref_model.language_model.gradient_checkpointing_enable()
                mm2t_ref_per_token_logps, t2i_ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    mm2t_input_ids=mm2t_prompt_completion_ids,
                    mm2t_images_seq_mask=mm2t_images_seq_mask,
                    mm2t_pixel_values=mm2t_pixel_values,
                    mm2t_images_emb_mask=mm2t_images_emb_mask,
                    mm2t_attention_mask=mm2t_attention_mask,
                    mm2t_logits_to_keep=mm2t_logits_to_keep,

                    t2i_inputs_ids=t2i_prompt_ids,
                    t2i_attention_mask=t2i_prompt_mask,
                    t2i_discrete_img_ids=t2i_completion_ids,
                    t2i_logits_to_keep=t2i_logits_to_keep,
                )
            else:  # for peft
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    mm2t_ref_per_token_logps, t2i_ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        mm2t_input_ids=mm2t_prompt_completion_ids,
                        mm2t_images_seq_mask=mm2t_images_seq_mask,
                        mm2t_pixel_values=mm2t_pixel_values,
                        mm2t_images_emb_mask=mm2t_images_emb_mask,
                        mm2t_attention_mask=mm2t_attention_mask,
                        mm2t_logits_to_keep=mm2t_logits_to_keep,

                        t2i_inputs_ids=t2i_prompt_ids,
                        t2i_attention_mask=t2i_prompt_mask,
                        t2i_discrete_img_ids=t2i_completion_ids,
                        t2i_logits_to_keep=t2i_logits_to_keep,
                    )

        return {
            "mm2t_prompt_completion_ids": mm2t_prompt_completion_ids,
            "mm2t_attention_mask": mm2t_attention_mask,
            "mm2t_logits_to_keep": mm2t_logits_to_keep,
            # loss
            "mm2t_completion_mask": mm2t_completion_mask,
            "mm2t_old_per_token_logps": mm2t_old_per_token_logps,
            "mm2t_ref_per_token_logps": mm2t_ref_per_token_logps,
            "mm2t_advantages": mm2t_advantages,
            # image
            "mm2t_pixel_values": mm2t_pixel_values,
            "mm2t_images_seq_mask": mm2t_images_seq_mask,
            "mm2t_images_emb_mask": mm2t_images_emb_mask,

            "t2i_inputs_ids": t2i_prompt_ids,
            "t2i_attention_mask": t2i_prompt_mask,
            "t2i_discrete_img_ids": t2i_completion_ids,
            "t2i_logits_to_keep": t2i_logits_to_keep,
            # loss
            "t2i_completion_mask": t2i_completion_mask,
            "t2i_old_per_token_logps": t2i_old_per_token_logps,
            "t2i_ref_per_token_logps": t2i_ref_per_token_logps,
            "t2i_advantages": t2i_advantages,
            "unified_advantages":unified_advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mm2t_per_token_logps, t2i_per_token_logps = self._get_per_token_logps(
            model,
            mm2t_input_ids=inputs["mm2t_prompt_completion_ids"],
            mm2t_attention_mask=inputs["mm2t_attention_mask"],
            mm2t_images_seq_mask=inputs["mm2t_images_seq_mask"],
            mm2t_pixel_values=inputs["mm2t_pixel_values"],
            mm2t_images_emb_mask=inputs["mm2t_images_emb_mask"],
            mm2t_logits_to_keep=inputs["mm2t_logits_to_keep"],

            t2i_inputs_ids=inputs["t2i_inputs_ids"],
            t2i_attention_mask=inputs["t2i_attention_mask"],
            t2i_discrete_img_ids=inputs["t2i_discrete_img_ids"],
            t2i_logits_to_keep=inputs["t2i_logits_to_keep"],
        )

        # Compute the loss
        mm2t_completion_mask = inputs["mm2t_completion_mask"]
        t2i_completion_mask = inputs["t2i_completion_mask"]

        if self.num_iterations > 1:
            mm2t_old_per_token_logps = inputs["mm2t_old_per_token_logps"]
            t2i_old_per_token_logps = inputs["t2i_old_per_token_logps"]
        else:
            mm2t_old_per_token_logps = mm2t_per_token_logps.detach()
            t2i_old_per_token_logps = t2i_per_token_logps.detach()

        mm2t_advantages = inputs["mm2t_advantages"]
        t2i_advantages = inputs["t2i_advantages"]
        unified_advantages = inputs["unified_advantages"]
        if self.task_args.unify_advantage and mm2t_advantages is not None and t2i_advantages is not None:
            unified_advantages = t2i_advantages + 0.8 * mm2t_advantages

            mm2t_per_token_loss = unified_advantages.unsqueeze(1) * (
                    mm2t_per_token_logps - mm2t_old_per_token_logps  # [n, L]
            ).exp()
            t2i_per_token_loss = unified_advantages.unsqueeze(1) * (
                    t2i_per_token_logps - t2i_old_per_token_logps  # [n, 576]
            ).exp()
        elif self.task_args.unify_reward and unified_advantages is not None:
            mm2t_per_token_loss = unified_advantages.unsqueeze(1) * (
                    mm2t_per_token_logps - mm2t_old_per_token_logps  # [n, L]
            ).exp()
            t2i_per_token_loss = unified_advantages.unsqueeze(1) * (
                    t2i_per_token_logps - t2i_old_per_token_logps  # [n, 576]
            ).exp()
        else:
            mm2t_per_token_loss = mm2t_advantages.unsqueeze(1) * (
                    mm2t_per_token_logps - mm2t_old_per_token_logps  # [n, L]
            ).exp()
            t2i_per_token_loss = t2i_advantages.unsqueeze(1) * (
                    t2i_per_token_logps - t2i_old_per_token_logps  # [n, 576]
            ).exp()

        if self.beta != 0.0:
            mm2t_log_ratio = inputs["mm2t_ref_per_token_logps"] - mm2t_per_token_logps
            mm2t_per_token_kl = mm2t_log_ratio.exp() - mm2t_log_ratio - 1.0

            mm2t_per_token_loss = -(mm2t_per_token_loss - self.beta * mm2t_per_token_kl)

            # Log the KL metric
            mm2t_mean_kl = ((mm2t_per_token_kl * mm2t_completion_mask).sum(
                dim=1) / mm2t_completion_mask.sum(dim=1)).mean()
            self._metrics["kl_mm2t"].append(
                self.accelerator.gather_for_metrics(mm2t_mean_kl).mean().item()
            )

            t2i_log_ratio = inputs["t2i_ref_per_token_logps"] - t2i_per_token_logps
            t2i_per_token_kl = t2i_log_ratio.exp() - t2i_log_ratio - 1.0

            t2i_per_token_loss = -(t2i_per_token_loss - self.beta * t2i_per_token_kl)

            t2i_mean_kl = ((t2i_per_token_kl * t2i_completion_mask).sum(
                dim=1) / t2i_completion_mask.sum(dim=1)).mean()
            self._metrics["kl_t2i"].append(
                self.accelerator.gather_for_metrics(t2i_mean_kl).mean().item()
            )
        else:
            mm2t_per_token_loss = -mm2t_per_token_loss
            t2i_per_token_loss = -t2i_per_token_loss

        # total loss
        loss_mm2t = ((mm2t_per_token_loss * mm2t_completion_mask).sum(
            dim=1) / mm2t_completion_mask.sum(dim=1)).mean()
        loss_t2i = ((t2i_per_token_loss * t2i_completion_mask).sum(
            dim=1) / t2i_completion_mask.sum(dim=1)).mean()

        return loss_mm2t + loss_t2i

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move
    # to device. Since we preprocess the data in `compute_loss`,
    # we need to override this method to skip this step.
    def _prepare_inputs(self, inputs):
        return self._generate_and_score_completions(inputs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)

        self._metrics.clear()
