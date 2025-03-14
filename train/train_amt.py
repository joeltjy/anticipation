import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = 'velocity-amt'
import sys

from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

from maestro_dataset import MaestroDataset

GPT2_MODEL_NAME = "stanford-crfm/music-medium-800k"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../maestrovelocity")
CACHE_DIR = os.path.expanduser("~/.cache")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "../checkpoint/velocity")
SEQLEN = 1024
LR = 6E-4
VELOCITY_OFFSET = 55028
VOCAB_SIZE = VELOCITY_OFFSET + 256


if __name__ == "__main__":
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)
    
    new_tokens = tokenizer.add_tokens(range(VELOCITY_OFFSET, VOCAB_SIZE))
    print("added tokens:", new_tokens[:5])

    model.resize_token_embeddings(VOCAB_SIZE)

    tokenizer.save_pretrained("velocity_amt_tokenizer")
    model.save_pretrained("velocity_amt_model")
    
    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    ds_train = MaestroDataset(
        DATA_DIR,
        split="train",
    )

    ds_valid = MaestroDataset(
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
        run_name="velocity-amt-1",
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