import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = 'base-amt'
import sys

from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

from lakh_dataset import LakhDataset

GPT2_MODEL_NAME = "stanford-crfm/music-medium-800k"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../lakhdata")
CACHE_DIR = os.path.expanduser("~/.cache")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "../checkpoint")
SEQLEN = 1024
LR = 6E-4


if __name__ == "__main__":
    # Load model
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)

    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    ds_train = LakhDataset(
        DATA_DIR,
        split="train",
    )

    ds_valid = LakhDataset(
        DATA_DIR,
        split="valid",
    )


    optimizer = AdamW(model.parameters(), lr=LR)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        max_steps=5000,
        save_steps=500,
        logging_dir="./logs",
        eval_steps=200,
        logging_steps=10,
        bf16=True,  # Enable mixed precision
        report_to="wandb",
        run_name="amt-medium-aug",
        dataloader_num_workers=4,
        do_eval=True,
        evaluation_strategy="steps",
        gradient_accumulation_steps=4,
        save_safetensors=False,
        # eval_on_start=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        optimizers=(optimizer, None),
    )

    # Train
    trainer.train()