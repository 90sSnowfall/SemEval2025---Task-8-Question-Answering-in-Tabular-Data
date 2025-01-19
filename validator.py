import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from fine_tuner import QADataset, collate_fn  # Assuming fine_tuner.py is in the same directory


def validate_model(model, tokenizer, dev_dataloader, device):
    """
    Validate the fine-tuned GPT-2 model on the validation dataset.

    Args:
        model (GPT2LMHeadModel): Fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
        dev_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run validation on (CPU/GPU).

    Returns:
        float: Average loss on the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    print("Starting validation...")
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(dev_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dev_dataloader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def main():
    # Path to the fine-tuned model and tokenizer
    fine_tuned_model_dir = "fine_tuned_models/epoch_10"  # Replace with the desired epoch folder
    dev_path = "preprocessed/qa_dev_gpt2.parquet"

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_dir)
    model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the validation dataset
    print("Loading validation dataset...")
    df_dev = pd.read_parquet(dev_path)
    dev_dataset = QADataset(df_dev, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_fn)

    # Validate the model
    validate_model(model, tokenizer, dev_dataloader, device)


if __name__ == "__main__":
    main()
