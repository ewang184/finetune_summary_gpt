from news_datasets import CNN_DailyMail_Dataset
from utils import get_random_percentage_subset
from model_def import FineTuner
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
import torch.distributed as dist

train_dataset = CNN_DailyMail_Dataset(split="train", tokenizer_name="facebook/bart-base")
val_dataset = CNN_DailyMail_Dataset(split="validation", tokenizer_name="facebook/bart-base")

train_size = len(train_dataset)
print(f'train_size is {train_size}')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)

batch_size = 16
accumulate_grad_batches = 4
num_epochs = 5
warmup_ratio = 0.1  # 10% of total steps

# Calculate total training steps
num_training_batches = len(train_loader)  # Total batches per epoch
effective_batches_per_epoch = num_training_batches // accumulate_grad_batches
total_steps = effective_batches_per_epoch * num_epochs

# Calculate warmup steps
warmup_steps = int(warmup_ratio * total_steps)

model = FineTuner(model_name="facebook/bart-base", lr=5e-5, warmup_steps=warmup_steps, total_steps=total_steps)

model.print_trainable_parameters()


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Replace with your desired ROUGE metric (e.g., rougeL or rouge2)
    mode="min",            # Maximize the ROUGE score
    filename="best-checkpoint-{epoch:02d}-{val_rougeL:.2f}",
    save_top_k=3,          # Save only the best checkpoint
    verbose=True,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Monitor ROUGE validation metric
    mode="min",            # Stop if the ROUGE score stops improving
    patience=3,            # Number of epochs with no improvement before stopping
    verbose=True,
)

trainer = Trainer(
    max_epochs=num_epochs,
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    accumulate_grad_batches=accumulate_grad_batches,
    enable_progress_bar=True,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
    gradient_clip_val=1.0,
)

trainer.fit(model, train_loader, val_loader)

trainer.save_checkpoint("model_checkpoint.ckpt")