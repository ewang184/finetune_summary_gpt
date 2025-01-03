import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from evaluate import load
from transformers import AutoTokenizer, AddedToken
import os 

class FineTuner(pl.LightningModule):
    def __init__(self, model_name="facebook/bart-base", lr=5e-5):
        super(FineTuner, self).__init__()
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if "facebook/bart" in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32, 
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none", 
            task_type="SEQ_2_SEQ_LM",
        )

        self.model = get_peft_model(model, lora_config)

        self.generation_config = model.generation_config

        self.lr = lr

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            eos_token_str = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
            self.tokenizer.add_special_tokens({'pad_token': AddedToken(eos_token_str)})

        self.rouge = load("rouge")

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)

        outputs = self(input_ids, attention_mask)
        loss = outputs.loss

        self.log("test_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        labels = batch["labels"].squeeze(1)

        outputs = self(input_ids, attention_mask)
        loss = outputs.loss

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=2,
        )

        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_results = self.rouge.compute(predictions=preds, references=refs)

        rouge1 = rouge_results["rouge1"]
        rouge2 = rouge_results["rouge2"]
        rougeL = rouge_results["rougeL"]

        self.log("val_rouge1", rouge1, prog_bar=True, sync_dist=True)
        self.log("val_rouge2", rouge2, prog_bar=True, sync_dist=True)
        self.log("val_rougeL", rougeL, prog_bar=True, sync_dist=True)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return {"val_loss": loss, "val_rouge1": rouge1, "val_rouge2": rouge2, "val_rougeL": rougeL}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def generate(self, input_ids, **kwargs):
        inputs = self.prepare_inputs_for_generation(input_ids, **kwargs)
        return self.model.generate(**inputs)