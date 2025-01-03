import torch
from transformers import AutoTokenizer, AddedToken
import pytorch_lightning as pl
from model_def import FineTuner
from torch.utils.data import DataLoader
from news_datasets import CNN_DailyMail_Dataset

# Load the checkpoint
checkpoint_path = "/teamspace/studios/this_studio/lightning_logs/version_11/checkpoints/best-checkpoint-epoch=06-val_rougeL=0.20.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu") 

epoch = checkpoint['epoch']
global_step = checkpoint['global_step']
pytorch_lightning_version = checkpoint['pytorch-lightning_version']
state_dict = checkpoint['state_dict']  # The model weights
optimizer_states = checkpoint['optimizer_states']
lr_schedulers = checkpoint['lr_schedulers']
loops = checkpoint['loops']
callbacks = checkpoint['callbacks']

# Instantiate the model
model_name = "facebook/bart-base"
model = FineTuner(model_name=model_name, lr=5e-5)

# Load the model weights from the checkpoint's state_dict
model.load_state_dict(state_dict)

# Now you can test the model or continue training from this checkpoint
print("Model loaded successfully!")

# Put the model in evaluation mode
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    eos_token_str = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    tokenizer.add_special_tokens({'pad_token': AddedToken(eos_token_str)})


model.generation_config.pad_token_id = tokenizer.pad_token_id

# Instantiate the dataset
dataset = CNN_DailyMail_Dataset(split="train", tokenizer_name="facebook/bart-base", max_input_length=256, max_target_length=128)

# Test loading a single sample
sample = dataset[0]

inputs = sample

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# If it's a single sample, add a batch dimension (making it a 2D tensor)
if input_ids.dim() == 1:
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

# Generate output
output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Input text:", input_text)

# Decode and print the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", output_text)
