"""
==============================
TRAIN
==============================

Erik Sarriegui Perez, AuriLab, Feb 2025
"""
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

import huggingface_hub

import torch
import os

from callback import PushToHubCallback
from dataset import loadFullCorpus
from utils import split_tokenize_count

def train_model() -> None:
    """
    Entrena un modelo GPT-2 en un dataset combinado de Latxa y Wikipedia en español.

    Esta función configura y ejecuta el entrenamiento de un modelo GPT-2 para tareas
    de lenguaje.  Utiliza el dataset combinado Latxa y Wikipedia, y emplea la librería
    `transformers` de Hugging Face.  Además, implementa funcionalidades como el uso
    de múltiples GPUs, compilación del modelo (si es posible), y el envío de
    checkpoints a Hugging Face Hub.

    ==========================
    SET UP ENVIROMENT
    ==========================
    """
    huggingface_hub.login(token = "<token>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    


    """
    ==========================
    DEFINING THE TRAINING
    ==========================
    """
    TOKENIZER_ID = "AuriLab/gpt-bi"
    PUSH_REPO_ID = "gpt-bi-erik"
    PUSH_ORGANIZATION = "AuriLab"
    PUSH_STEPS = 10000

    EPOCHS = 1
    OUTPUT_DIR = "./gpt-bi-pretrained"
    BLOCK_SIZE = 1024
    BATCH_SIZE = 2
    LEARNING_RATE = 6e-4
    WEIGHT_DECAY = 0.1
    MAX_GRAD_NORM = 1.0
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_STEPS = 2000
    DDP_FIND_UNUSED_PARAMETERS = False
    GRADIENT_CHECKPOINTING = False
    NUM_WORKERS = 8
    EVAL_SAVE_LOGGING_STRATEGY = "steps"
    REPORT = "none"
    
    
    """
    ==========================
    MODEL & TOKENIZER
    ==========================
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_ID)

    config = GPT2Config(
        vocab_size = 32003,
        n_positions = BLOCK_SIZE,
        n_ctx = BLOCK_SIZE,
        n_embd = 768,
        n_layer = 12,
        n_head = 12,
    )

    model = GPT2LMHeadModel(config).to(device)


    """
    ==========================
    DATASET
    ==========================
    """
    dataset = loadFullCorpus()
    
    tokenized_train_dataset = dataset["train"].map(
        split_tokenize_count,
        batched = True,
        num_proc = NUM_WORKERS,
        remove_columns="text",
        fn_kwargs = {"tokenizer": tokenizer, "block_size": BLOCK_SIZE}
    )

    tokenized_test_dataset = dataset["test"].map(
        split_tokenize_count,
        batched = True,
        num_proc = NUM_WORKERS,
        remove_columns="text",
        fn_kwargs = {"tokenizer": tokenizer, "block_size": BLOCK_SIZE}
    )

    n_train_tokens = sum(tokenized_train_dataset['n_tokens'])
    n_test_tokens = sum(tokenized_test_dataset['n_tokens'])

    tokenized_train_dataset = tokenized_train_dataset.remove_columns("n_tokens")
    tokenized_test_dataset = tokenized_test_dataset.remove_columns("n_tokens")

    """
    ==========================
    LAST CHECKS
    ==========================
    """
    dataCollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm = False
    )

    callback_push = PushToHubCallback(PUSH_REPO_ID, PUSH_ORGANIZATION, PUSH_STEPS)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert BATCH_SIZE % world_size == 0, f"Batch size {BATCH_SIZE} no es divisible por el número de GPUs {world_size}"



    """
    ==========================
    TRAINING ARGUMENTS
    ==========================
    """
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        overwrite_output_dir = True,
        remove_unused_columns=False,
                
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = int(BATCH_SIZE / world_size),
                
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        max_grad_norm = MAX_GRAD_NORM,
        lr_scheduler_type = LR_SCHEDULER_TYPE,
        warmup_steps = WARMUP_STEPS,
                
        fp16 = True,
                
        save_strategy = EVAL_SAVE_LOGGING_STRATEGY,
        save_steps = PUSH_STEPS,
        logging_strategy = EVAL_SAVE_LOGGING_STRATEGY,
        logging_steps = PUSH_STEPS,
        eval_strategy = EVAL_SAVE_LOGGING_STRATEGY,
                
        ddp_find_unused_parameters = DDP_FIND_UNUSED_PARAMETERS,
        report_to = REPORT,
                
        gradient_checkpointing = GRADIENT_CHECKPOINTING
    )
            
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_test_dataset,
        data_collator = dataCollator,
        callbacks = [callback_push]
    )
    


    """
    ==========================
    PRE-TRAIN LOGS
    ==========================
    """
    print(f"[INFO] Dataset cargado y tokenizado, {round(n_train_tokens/1_000_000, 2)}M tokens en entrenamiento y {round(n_test_tokens/1_000_000)}M tokens en prueba")
    print(f"[INFO] Iniciando el entrenamiento en {device}")

    if device == "cuda":
        print(f"[INFO] El entrenamiento se realizará con {world_size} GPUs")
    
    else:
        print("[INFO] El entrenamiento se realizará en CPU, utiliza GPU (CUDA) para acelerar el proceso, CUDA goes BRRRRRRR")

    """
    ==========================
    TRAIN
    ==========================
    """        
    trainer.train()
    model.push_to_hub(repo_id = PUSH_REPO_ID, organization = PUSH_ORGANIZATION)



if __name__ == "__main__":
    """
    ==========================
    LAUNCH TRAINING
    ==========================
    """
    train_model()