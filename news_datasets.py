import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AddedToken

class CNN_DailyMail_Dataset(Dataset):
    def __init__(self, split="train", tokenizer_name="facebook/bart-base", max_input_length=500, max_target_length=256):
        """
        Args:
            split (str): Which split to load ("train", "validation", or "test").
            tokenizer_name (str): Pre-trained tokenizer name from Hugging Face.
            max_input_length (int): Maximum length of input article.
            max_target_length (int): Maximum length of target summary.
        """
        self.dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            eos_token_str = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
            self.tokenizer.add_special_tokens({'pad_token': AddedToken(eos_token_str)})

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.split = split

    def preprocess_text(self, article, highlights):
        article = article.replace("\n", " ").strip()
        highlights = highlights.replace("\n", " ").strip()
        return article, highlights

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single example from the dataset.
        """
        sample = self.dataset[idx]
        
        article, highlights = self.preprocess_text(sample["article"], sample["highlights"])

        inputs = self.tokenizer(
            article, 
            max_length=self.max_input_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )

        labels = self.tokenizer(
            highlights, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )

        inputs = {key: value.squeeze() for key, value in inputs.items()}
        labels = {key: value.squeeze() for key, value in labels.items()}

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
        }