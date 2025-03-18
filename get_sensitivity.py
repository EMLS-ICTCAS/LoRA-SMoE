# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

dataset = ['boolq', 'piqa','social_i_qa','winogrande','ARC-Easy','ARC-Challenge','openbookqa','hellaswag']
DATAPATH = r'../dataset/'+dataset[2]+'/train.json'
# MAX_STEPS = 36,72,144,288 Integer multiples of model layers
MAX_STEPS = 288
Toutput_dir = r'../dataset/'+dataset[2]+'/T3B_SENTI288.json'

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import json
from typing import List
from badam import BlockOptimizer
import fire
import torch
import transformers
# from datasets import load_dataset
from datasets import load_dataset, concatenate_datasets
from typing import List, Optional, Union
from transformers import TrainerState,TrainingArguments,TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
sys.path.append("/home/xjz/proj/Subspace-Tuning/CR_MR/peft/src/")
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402

# model/data params
base_model: str = "/home/xjz/model/Qwen2_5_3B"  # the only required argument
data_path: str = DATAPATH
output_dir: str = ""
adapter_name: str = "lora"
load_8bit : bool = False
# training hyperparams
batch_size:int = 1
micro_batch_size:int =1
num_epochs: int = 1.2
learning_rate: float = 1e-5
weight_decay: float = 0.0
cutoff_len: int = 256
val_set_size: int = 0
use_gradient_checkpointing: bool = False
eval_step: int = 0
save_step: int = 3000
# lora hyperparams
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj"]
# bottleneck adapter hyperparams
bottleneck_size: int = 256
non_linearity: str = "tanh"
adapter_dropout: float = 0.0
use_parallel_adapter: bool = False
use_adapterp: bool = False
target_modules: List[str] = None
# Dora hyperparams
dora_simple: bool = True
Wdecompose_target_modules: List[str] = None
scaling: Union[float, str] = 1.0
# prefix tuning hyperparams
num_virtual_tokens: int = 30
# llm hyperparams
train_on_inputs: bool = True  # if False, masks out inputs in loss
group_by_length: bool = False  # faster, but produces an odd training loss curve
# wandb params
wandb_project: str = ""
wandb_run_name: str = ""
wandb_watch: str = "" # options: false | gradients | all
wandb_log_model: str = ""  # options: false | true
resume_from_checkpoint: str = None  # either training checkpoint or final adapter

gradient_accumulation_steps = int(batch_size) // int(micro_batch_size)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps / world_size

# Check if parameter passed or if set within environ
use_wandb = len(wandb_project) > 0 or (
    "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
)
# Only overwrite environ if wandb param passed
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
if len(wandb_watch) > 0:
    os.environ["WANDB_WATCH"] = wandb_watch
if len(wandb_log_model) > 0:
    os.environ["WANDB_LOG_MODEL"] = wandb_log_model

if load_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
    )
if model.config.model_type == "llama":
# Due to the name of transformers' LlamaTokenizer, we have to do this
# need to handle llama 3 separately
    if "Llama-3" in base_model:
        print("load llama-3 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
print(model)



tempgrad_dict = {}
import pickle
class SkipStepCallback(transformers.TrainerCallback):
    grad_dict = {}
    has_grad = 0
    skip_kwd_list = ['input_layernorm', 'post_attention_layernorm', 'bias','lm_head','embed_tokens','rotary_emb','act_fn']
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return None
    
    def on_step_begin(self, args, state, control, **kwargs):
        pass
        
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not any(kwd in name for kwd in self.skip_kwd_list):
                    self.grad_dict[name] += (param.grad**2).sum().item()
        self.has_grad = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        model =  kwargs['model']
        grad_skip_kwd_list = ['head', 'norm', 'bias','input_layernorm','post_attention_layernorm','input_layernorm', 'bias','lm_head','embed_tokens','rotary_emb','act_fn']  # Fully tune head and class token, freeze patch_embed,
        for name, param in model.named_parameters():
            if not any(kwd in name for kwd in grad_skip_kwd_list):
                self.grad_dict[name] = 0.0
        print(self.grad_dict)
        
    def on_train_end(self, args, state, control, **kwargs):
        json_lines = []
        for key, value in self.grad_dict.items():
            json_line = json.dumps({key: value})
            json_lines.append(json_line)
        print(json_lines)
 
        with open(Toutput_dir, 'w') as f:
            for line in json_lines:
                f.write(line + '\n')
        print("JSON file saved successfully.")

if data_path.endswith(".json"):  # todo: support jsonl
    data = load_dataset("json", data_files=data_path)
# data2 = load_dataset("json", data_files='/home/xjz/proj/Subspace-Tuning/CR_MR/math_10k.json')
# data = concatenate_datasets([data['train'], data2['train']])
else:
    data = load_dataset(data_path)

def process_func(example):
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id] 
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1 
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > cutoff_len:
        input_ids = input_ids[:cutoff_len]   
        attention_mask = attention_mask[:cutoff_len]    
        labels = labels[:cutoff_len] 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if val_set_size > 0:
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(process_func,remove_columns=data['train'].column_names)
    )
    val_data = (
        train_val["test"].shuffle().map(process_func,remove_columns=data['test'].column_names)
    )
else:
    train_data = data['train'].shuffle().map(process_func,remove_columns=data['train'].column_names)
    val_data = None
    
original_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler. ExponentialLR(original_optimizer, gamma=0.95)
optimizer = BlockOptimizer(
base_optimizer=original_optimizer, 
named_parameters_list=list(model.named_parameters()), 
switch_block_every=1, 
switch_mode="descending",
verbose=2 # information level, will print trainable parameters when setting to 2
)
trainer = transformers.Trainer(
model=model,
train_dataset=train_data,
eval_dataset=val_data,

args=transformers.TrainingArguments(
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=20,
    num_train_epochs=num_epochs,
    max_steps = MAX_STEPS,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    # fp16=True,
    logging_steps=20,
    # 测试有用没
    logging_dir= output_dir,
    optim="adamw_torch",
    evaluation_strategy="steps" if val_set_size > 0 else "no",
    eval_steps=eval_step if val_set_size > 0 else None,
    save_strategy="no",
    save_steps=save_step,
    output_dir=output_dir,
    save_total_limit=3,
    load_best_model_at_end=True if val_set_size > 0 else False,
    ddp_find_unused_parameters=False if ddp else None,
    group_by_length=group_by_length,
    report_to="wandb" if use_wandb else None,
    run_name=wandb_run_name if use_wandb else None,
),
data_collator=transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
),
optimizers=(optimizer, scheduler),

)
trainer.add_callback(SkipStepCallback)
model.config.use_cache = False
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)
trainer.train(resume_from_checkpoint=resume_from_checkpoint)


