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

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

from transformers import TrainerCallback
from trl import (
    GRPOConfig, ModelConfig, ScriptArguments,
    TrlParser, get_peft_config
)

# from corl.open_r1.trainer.grpo_trainer import JanusProJointGRPOTrainer
from corl.open_r1.trainer.grpo_trainer_unified import JanusProUnifiedGRPOTrainer
# from corl.open_r1.trainer.grpo_trainer_unified_v2 import JanusProUnifiedGRPOTrainer
from corl.open_r1.rewards import reward_funcs_registry


class ParameterInfoCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Fixed: {name}, Shape: {param.shape}, Parameters: {param.numel():,}")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'",
            "nargs": "+",
        },
    )
    # for janus
    delete_unused_module: bool = field(
        default=False,
        metadata={"help": "Whether to delete unused modules in janus."},
    )
    save_oom: bool = field(
        default=False,
        metadata={"help": "Whether to delete unused modules in janus."},
    )
    task_format: Optional[str] = field(
        default="t2i",
        metadata={
            "help": "Possible values: 't2i' (text to image), 'mm2t' (mm to text), "
                    "'joint': , 'unify'"
        },
    )
    mm2t_format: Optional[str] = field(
        default='qa',
        metadata={
            "help": "Possible values: 'qa', 'od' (object detection), 'oc' (object classification)"
        },
    )
    unify_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to delete unused modules in janus."},
    )
    unify_reward: bool = field(
        default=True,
        metadata={"help": "Whether to delete unused modules in janus."},
    )
    model_ckpt_dir: str = field(
        default="playground/checkpoint/"
    )
    blip_model_ckpt: str = field(
        default="blip-image-captioning-base"
    )
    dataset_cache_dir: str = field(
        default="playground/data/cache/"
    )


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        cache_dir=script_args.dataset_cache_dir,
    )

    # # for debug
    # sampled_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    # dataset["train"] = sampled_dataset

    # Format into conversation
    def make_conversation_t2i(example):
        return {
            "prompt": [
                {
                    "role": "<|User|>",
                    "content": f"{example['prompt'].strip()}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        }

    def make_conversation_mm2t(example):
        if script_args.mm2t_format == 'qa':
            question = example["qa_problem"]
        elif script_args.mm2t_format == 'oc':
            question = example["cls_problem"]
        elif script_args.mm2t_format == 'od':
            question = example["od_problem"]
        else:
            question = example["problem"]

        return {
            "qa_prompt": [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    # "images": [example["image"]],
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
        }

    def make_conversation_joint(example):
        return {
            "prompt": [
                {
                    "role": "<|User|>",
                    "content": f"{example['prompt'].strip()}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ],
            "qa_prompt": [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{example['qa_problem']}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        }

    if script_args.task_format == "t2i":
        dataset = dataset.map(make_conversation_t2i)
    elif script_args.task_format == "mm2t":
        dataset = dataset.map(make_conversation_mm2t)
        # dataset = dataset.remove_columns(["type"])
    else:
        dataset = dataset.map(make_conversation_joint)

    trainer_cls = JanusProUnifiedGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,

        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[
            script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,

        peft_config=get_peft_config(model_args),
        # callbacks=[ParameterInfoCallback()],

        # for janus
        task_args=script_args,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))

    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
