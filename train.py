from news_datasets import CNN_DailyMail_Dataset
from utils import get_random_percentage_subset
from model_def import GPT2FineTuner
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist

train_dataset = get_random_percentage_subset(CNN_DailyMail_Dataset(split="train"),1)
val_dataset = get_random_percentage_subset(CNN_DailyMail_Dataset(split="validation"),1)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

model = GPT2FineTuner(model_name="gpt2", lr=5e-5)

model.print_trainable_parameters()


checkpoint_callback = ModelCheckpoint(
    monitor="val_rougeL",  # Replace with your desired ROUGE metric (e.g., rougeL or rouge2)
    mode="max",            # Maximize the ROUGE score
    filename="best-checkpoint-{epoch:02d}-{val_rougeL:.2f}",
    save_top_k=1,          # Save only the best checkpoint
    verbose=True,
)

early_stopping_callback = EarlyStopping(
    monitor="val_rougeL",  # Monitor ROUGE validation metric
    mode="max",            # Stop if the ROUGE score stops improving
    patience=3,            # Number of epochs with no improvement before stopping
    verbose=True,
)

trainer = Trainer(
    max_epochs=10,
    devices=3,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    enable_progress_bar=True,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback]#, early_stopping_callback],
)

trainer.fit(model, train_loader, val_loader)

trainer.save_checkpoint("model_checkpoint.ckpt")