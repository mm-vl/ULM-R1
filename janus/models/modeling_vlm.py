# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)
        #
        self.hidden_size = self.language_config.hidden_size


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            images_seq_mask: torch.LongTensor,
            images_emb_mask: torch.LongTensor,
            **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        # input_ids[input_ids < 0] = 0  # ignore the image embeddings !!! in-place operation
        # inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  # torch.bfloat16
        safe_input_ids = input_ids.masked_fill(input_ids < 0, 0)
        inputs_embeds = self.language_model.get_input_embeddings()(safe_input_ids)

        # replace with the image embeddings
        # inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]
        new_inputs_embeds = inputs_embeds.clone()
        replacement = images_embeds[images_emb_mask]
        new_inputs_embeds = new_inputs_embeds.masked_scatter(
            images_seq_mask.unsqueeze(-1).expand_as(new_inputs_embeds),
            replacement.reshape(-1)
        )

        return new_inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    def encode_und_img_embeds(self, pixel_values: torch.FloatTensor,):
        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)

        return images_embeds

    def forward(
            self,
            # for und.
            mm2t_input_ids: torch.LongTensor = None,
            mm2t_attention_mask: Optional[torch.Tensor] = None,
            mm2t_inputs_embeds: Optional[torch.FloatTensor] = None,
            mm2t_pixel_values: Optional[torch.FloatTensor] = None,
            mm2t_images_seq_mask: Optional[torch.LongTensor] = None,
            mm2t_images_emb_mask: Optional[torch.LongTensor] = None,
            mm2t_logits_to_keep: Union[int, torch.Tensor] = 0,

            # for t2i
            t2i_input_ids: torch.LongTensor = None,
            t2i_attention_mask: Optional[torch.Tensor] = None,
            t2i_inputs_embeds: Optional[torch.FloatTensor] = None,
            t2i_discrete_img_ids = None,
            t2i_pixel_values: Optional[torch.FloatTensor] = None,
            t2i_logits_to_keep: Union[int, torch.Tensor] = 0,

            task="understanding",
            labels=None,
            return_dict=True,
            **kwargs
    ):
        if task == "understanding":
            if mm2t_inputs_embeds is None:
                mm2t_inputs_embeds = self.prepare_inputs_embeds(
                    input_ids=mm2t_input_ids,
                    images_seq_mask=mm2t_images_seq_mask,
                    pixel_values=mm2t_pixel_values,
                    images_emb_mask=mm2t_images_emb_mask,
                )
            return self.language_model(
                inputs_embeds=mm2t_inputs_embeds,
                attention_mask=mm2t_attention_mask,
                labels=labels,
                logits_to_keep=mm2t_logits_to_keep
            )

        elif task == "generation":
            if t2i_inputs_embeds is None:
                t2i_inputs_embeds = self.language_model.get_input_embeddings()(t2i_input_ids)

                if t2i_discrete_img_ids is None and t2i_pixel_values is not None:
                    with torch.inference_mode():
                        # all_labels
                        t2i_discrete_img_ids = self.gen_vision_model.encode(
                            t2i_pixel_values)[-1][-1]
                        t2i_discrete_img_ids = t2i_discrete_img_ids.reshape(
                            t2i_inputs_embeds.shape[0], -1)

                t2i_img_embeds = self.prepare_gen_img_embeds(t2i_discrete_img_ids)
                t2i_img_mask = t2i_discrete_img_ids >= 0

                t2i_inputs_embeds = torch.cat([t2i_inputs_embeds, t2i_img_embeds], dim=1)
                t2i_attention_mask = torch.cat([t2i_attention_mask, t2i_img_mask], dim=1)

            outputs = self.language_model.model(
                inputs_embeds=t2i_inputs_embeds,
                attention_mask=t2i_attention_mask,
                use_cache=False,
                past_key_values=None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.gen_head(hidden_states[:, -t2i_logits_to_keep:, :]).contiguous()

            return CausalLMOutputWithPast(
                # loss=loss,
                logits=logits,
                # logits=all_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                # hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        else:  # joint
            if mm2t_inputs_embeds is None:
                mm2t_inputs_embeds = self.prepare_inputs_embeds(
                    input_ids=mm2t_input_ids,
                    images_seq_mask=mm2t_images_seq_mask,
                    pixel_values=mm2t_pixel_values,
                    images_emb_mask=mm2t_images_emb_mask,
                )

            mm2t_logits = self.language_model(
                inputs_embeds=mm2t_inputs_embeds,
                attention_mask=mm2t_attention_mask,
                labels=labels,
                logits_to_keep=mm2t_logits_to_keep
            ).logits

            if t2i_inputs_embeds is None:
                t2i_inputs_embeds = self.language_model.get_input_embeddings()(t2i_input_ids)

                if t2i_discrete_img_ids is None and t2i_pixel_values is not None:
                    with torch.inference_mode():
                        # all_labels
                        t2i_discrete_img_ids = self.gen_vision_model.encode(
                            t2i_pixel_values)[-1][-1]
                        t2i_discrete_img_ids = t2i_discrete_img_ids.reshape(
                            t2i_inputs_embeds.shape[0], -1)

                t2i_img_embeds = self.prepare_gen_img_embeds(t2i_discrete_img_ids)
                t2i_img_mask = t2i_discrete_img_ids >= 0

                t2i_inputs_embeds = torch.cat([t2i_inputs_embeds, t2i_img_embeds], dim=1)
                t2i_attention_mask = torch.cat([t2i_attention_mask, t2i_img_mask], dim=1)

            outputs = self.language_model.model(
                inputs_embeds=t2i_inputs_embeds,
                attention_mask=t2i_attention_mask,
                use_cache=False,
                past_key_values=None
            )

            t2i_logits = self.gen_head(
                outputs.last_hidden_state[:, -t2i_logits_to_keep:, :]
            ).contiguous()

            return mm2t_logits, t2i_logits

    @torch.inference_mode()
    def t2i_generate_parallel(
            self,
            input_ids,  # [bs, len]
            attention_mask=None,  # [bs, len]
            # for t2i
            parallel_size: int = 16,  # equal to num_return_sequences, =1:
            temperature: float = 1,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,  # 24x24
            img_size: int = 384,
            patch_size: int = 16,
            pad_id: int = 100002,
            seed=42,
            completion_type="image",
            image_resize=False,
    ):
        device = input_ids.device
        bs, max_len = input_ids.size()

        # [bs * 2 * parallel_size, max_len]
        attention_mask = torch.cat([
            attention_mask, torch.ones((bs, image_token_num_per_image), device=device)
        ], dim=-1)

        attention_mask = attention_mask.repeat_interleave(2 * parallel_size, dim=0)
        tokens = input_ids.repeat_interleave(2 * parallel_size, dim=0)
        mask = torch.arange(bs * 2 * parallel_size, device=device) % 2 == 1  # unconditional input
        tokens[mask, 1:-1] = pad_id

        # [bs * 2 * parallel_size, max_len, dim]
        inputs_embeds = self.language_model.get_input_embeddings()(tokens)

        # if completion_type == "mm_sim":
        #     t2i_inputs_embeds = inputs_embeds.clone()[mask]  # [bs * parallel_size, max_len, dim]
        #     t2i_inputs_embeds = t2i_inputs_embeds[:, :-1, :]

        generated_tokens = torch.zeros(
            (bs * parallel_size, image_token_num_per_image), dtype=torch.int
        ).to(device)  # discrete image IDs
        for i in range(image_token_num_per_image):  # Autoregressive
            outputs = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            # CFG
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)

            img_embeds = self.prepare_gen_img_embeds(next_token)  # [2 * bs * parallel_size, dim]
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # completions
        completions = []
        if completion_type == "image":
            dec = self.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[bs * parallel_size, 8, img_size // patch_size, img_size // patch_size]
            )  # [bs * parallel_size, 3, 384, 384], torch.bfloat16,

            # ******* tensor *******
            if image_resize:
                dec_224 = F.interpolate(
                    dec.to(torch.float32), size=(224, 224),
                    mode="bilinear", align_corners=False
                )
                dec_224 = (255 * (dec_224 * 0.5 + 0.5)).clamp(0, 255)
                completions.extend([d for d in dec_224])
            else:
                # [bs * parallel_size, 3, 384, 384]
                # dec_384 = dec.to(torch.float32)
                # dec_384 = (255 * (dec_384 * 0.5 + 0.5)).clamp(0, 255)
                # completions.extend([d for d in dec_384])

                # ******* numpy *******
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                # [bs * parallel_size, 384, 384, 3]
                dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
                completions.extend([Image.fromarray(d) for d in dec])

        return generated_tokens, completions

    @torch.inference_mode()
    def t2i_generate_wo_cfg(
            self,
            input_ids,  # [bs, len]
            attention_mask=None,  # [bs, len]
            # for t2i
            parallel_size: int = 16,  # equal to num_return_sequences, =1:
            temperature: float = 1,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,  # 24x24
            img_size: int = 384,
            patch_size: int = 16,
            pad_id: int = 100002,
            seed=42,
            completion_type="mm_sim",
    ):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        device = input_ids.device
        bs, max_len = input_ids.size()

        # [bs * 2 * parallel_size, max_len]
        attention_mask = torch.cat([
            attention_mask, torch.ones((bs, image_token_num_per_image), device=device)
        ], dim=-1)

        # [bs * 2 * parallel_size, max_len]
        attention_mask = attention_mask.repeat_interleave(2 * parallel_size, dim=0)
        tokens = input_ids.repeat_interleave(2 * parallel_size, dim=0)
        mask = torch.arange(bs * 2 * parallel_size, device=device) % 2 == 1
        tokens[mask, 1:-1] = pad_id

        # [bs * 2 * parallel_size, max_len, dim]
        inputs_embeds = self.language_model.get_input_embeddings()(tokens)

        if completion_type == "mm_sim":
            t2i_inputs_embeds = inputs_embeds.clone()[mask]  # [bs * parallel_size, max_len, dim]
            t2i_inputs_embeds = t2i_inputs_embeds[:, :-1, :]

        generated_tokens = torch.zeros(
            (bs * parallel_size, image_token_num_per_image), dtype=torch.int
        ).to(device)  # discrete image IDs
        for i in range(image_token_num_per_image):  # Autoregressive
            outputs = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            # CFG
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)

            img_embeds = self.prepare_gen_img_embeds(next_token)  # [2 * bs * parallel_size, dim]
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # completions
        completions = {}
        if completion_type == "image":
            dec = self.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[bs * parallel_size, 8, img_size // patch_size, img_size // patch_size]
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            # [bs * parallel_size, 384, 384, 3]

            dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

            completions['generated_image'] = dec

        return generated_tokens, completions


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
