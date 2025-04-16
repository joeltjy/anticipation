import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = 'velocity-augment-amt'
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from transformers.trainer_callback import TrainerCallback
sys.path.append(os.path.abspath(".."))
from anticipation.sample import generate
from anticipation.convert import events_to_midi
from anticipation.vocab import MIDI_VOCAB_SIZE
from maestro_dataset import MaestroDataset

GPT2_MODEL_NAME = "stanford-crfm/music-medium-800k"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../orcd/c7/pool/maestrovelocityaugment")
CACHE_DIR = os.path.expanduser("~/.cache")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "../../orcd/c7/pool/checkpoint/velocityaugment")
SEQLEN = 1024
LR = 6E-4
VELOCITY_OFFSET = 55028
VOCAB_SIZE = VELOCITY_OFFSET + 256


SAVE_STEPS = 5
GENERATE_STEPS = 10
# Custom callback class
class CustomCallback(TrainerCallback):
    def __init__(self, model, custom_function, interval_steps=GENERATE_STEPS):
        self.model = model
        self.custom_function = custom_function
        self.interval_steps = interval_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval_steps == 0:
            self.custom_function(state.global_step, self.model)

if __name__ == "__main__":
    # Load model
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)
    
    model.resize_token_embeddings(VOCAB_SIZE)
    model.save_pretrained("velocity_augment_amt_model")
    
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
        save_steps=SAVE_STEPS,
        logging_dir="./logs",
        eval_steps=200,
        logging_steps=10,
        bf16=True,
        report_to="wandb",
        run_name="velocity-augment-amt-1",
        dataloader_num_workers=4,
        do_eval=True,
        evaluation_strategy="steps",
        gradient_accumulation_steps=4,
        save_safetensors=False,
    )
    
    def generate_sample_function(global_step, model, num_samples=1):

        print(f"Generating {num_samples} samples at step {global_step}")
        print("model", model)
        with torch.no_grad():
            for i in range(num_samples):
                sample = generate(model, start_time=0, end_time=20, top_p=.98, include_velocity=True)
                print("sample 1", sample)
                mid = events_to_midi(sample, include_velocity=True)
                mid.save(f'checkpoint-{global_step}-sample-{i}-velocity.mid')
        
        print(f"Generated {num_samples} samples at step {global_step}")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        optimizers=(optimizer, None),
        callbacks=[CustomCallback(model, generate_sample_function, interval_steps=GENERATE_STEPS)]
    )
    
    # Train
    trainer.train()