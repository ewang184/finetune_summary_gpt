from news_datasets import CNN_DailyMail_Dataset
from utils import get_random_percentage_subset
from model_def import GPT2FineTuner
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch

test_dataset = get_random_percentage_subset(CNN_DailyMail_Dataset(split="test"), 1)
test_loader = DataLoader(test_dataset, batch_size=2)

checkpoint_path = "/teamspace/studios/this_studio/lightning_logs/version_2/checkpoints/best-checkpoint-epoch=05-val_rougeL=0.14.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU

epoch = checkpoint['epoch']
global_step = checkpoint['global_step']
pytorch_lightning_version = checkpoint['pytorch-lightning_version']
state_dict = checkpoint['state_dict'] 
optimizer_states = checkpoint['optimizer_states']
lr_schedulers = checkpoint['lr_schedulers']
loops = checkpoint['loops']
callbacks = checkpoint['callbacks']

model = FineTuner(model_name="gpt2", lr=5e-5)

model.load_state_dict(state_dict)

print("Model loaded successfully!")

model.eval()

trainer = Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    enable_progress_bar=True,
)

results = trainer.test(model, test_loader)
print(results)
