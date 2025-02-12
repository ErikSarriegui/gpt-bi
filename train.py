from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.optim import AdamW

import huggingface_hub

import torch
import os

## -- PROJECT IMPORTS
from callback import PushToHubCallback
from dataset import LatxaDataset
from utils import calculate_steps

"""
==========================
SET UP ENVIROMENT
==========================
"""
device = "cuda" if torch.cuda.is_available() else "cpu"

huggingface_hub.login(token = os.getenv("HF_TOKEN"))

torch.backends.cudnn.benchmark = True

"""
==========================
HYPERPARAMETERS
==========================
"""

## -- Tokenizer
tokenizer_id = "AuriLab/gpt-bi"
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_id)

## -- Model
block_size = 1024
embedding_size = 768
transformer_layers = 12
number_heads = 12

## -- Training
batch_size = 32
gradient_accumulation_steps = 1
total_tokens = 4_000_000_000

max_steps = calculate_steps(
    batch_size = batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    total_tokens = total_tokens,
    block_size = block_size
)

## --Training Hyperparams
learning_rate = 6e-4
weight_decay = 0.1
max_grad_norm = 1.0
lr_scheduler_type = "cosine"
warmup_steps = 2000
save_steps = 2000
ddp_find_unused_parameters = False
gradient_checkpointing = True           # Esto a lo mejor se puede quitar
optimizer = AdamW()

"""
==========================
DATASET
==========================
"""
train_dataset = LatxaDataset(tokenizer, split = "train", block_size = block_size, batch_size = batch_size)
eval_dataset = LatxaDataset(tokenizer, split = "validation", block_size = block_size, batch_size = batch_size)

"""
==========================
MODEL
==========================
"""
config = GPT2Config(
    vocab_size = tokenizer.vocab_size,
    n_positions = block_size,
    n_ctx = block_size,
    n_embd = embedding_size,
    n_layer = transformer_layers,
    n_head = number_heads,
)

model = GPT2LMHeadModel(config)
model.to(device)

"""
==========================
TRAINING
==========================
"""
training_args = TrainingArguments(
    output_dir = "./gpt2-pretrained",
    overwrite_output_dir = True,
    
    # Parámetros de entrenamiento principales
    max_steps = max_steps,
    per_device_train_batch_size = batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    
    # Optimizador y learning rate
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    max_grad_norm = max_grad_norm,
    lr_scheduler_type = lr_scheduler_type,
    warmup_steps = warmup_steps,
    
    # Precisión y optimización
    bf16 = True,
    
    # Checkpoints y logging
    save_strategy = "steps",
    save_steps = save_steps,
    logging_strategy = "steps",
    logging_steps = 100,
    eval_strategy="steps",
    
    # Otros parámetros
    ddp_find_unused_parameters = ddp_find_unused_parameters,
    report_to = "none",
    
    # Optimizaciones de memoria
    gradient_checkpointing = gradient_checkpointing,
    optim = optimizer
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False,
    pad_to_multiple_of = 8
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
    callbacks = [PushToHubCallback()]
)

"""
==========================
TRAIN
==========================
"""
print("[INFO] Iniciando el entrenamiento")

trainer.train()
model.push_to_hub("gpt-bi", organization="AuriLab")