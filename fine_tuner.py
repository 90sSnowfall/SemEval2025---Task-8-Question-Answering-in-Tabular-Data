import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm


class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=1024):
        """
        Custom Dataset for Question-Answer pairs.

        Args:
            dataframe (pd.DataFrame): DataFrame containing 'question' and 'answer_tokens' columns.
            tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
            max_length (int): Maximum sequence length for tokenized inputs.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure all answer_tokens are lists
        self.data["answer_tokens"] = self.data["answer_tokens"].apply(
            lambda x: x if isinstance(x, list) else (x.tolist() if hasattr(x, "tolist") else [])
        )

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question_text = row["question"]
        answer_tokens = row["answer_tokens"]

        # Validate answer_tokens format
        if not isinstance(answer_tokens, list):
            raise ValueError(f"Invalid format for answer_tokens at index {idx}: {answer_tokens}. Must be a list.")

        # Encode question text
        question_tokens = self.tokenizer.encode(question_text, add_special_tokens=True)

        # Combine question and answer tokens
        input_ids = question_tokens + answer_tokens

        # Truncate to max length
        input_ids = input_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def fine_tune_gpt2(train_path, dev_path, output_dir="models", batch_size=4, epochs=3, lr=5e-5):
    """
    Fine-tune GPT-2 model on QA dataset.

    Args:
        train_path (str): Path to the preprocessed training dataset (parquet file).
        dev_path (str): Path to the preprocessed validation dataset (parquet file).
        output_dir (str): Directory to save fine-tuned model.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load datasets
    print("Loading datasets...")
    df_train = pd.read_parquet(train_path)
    df_dev = pd.read_parquet(dev_path)

    train_dataset = QADataset(df_train, tokenizer)
    dev_dataset = QADataset(df_dev, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_dataloader)}")

        # Save model after each epoch
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch + 1}"))
        tokenizer.save_pretrained(os.path.join(output_dir, f"epoch_{epoch + 1}"))

    print("Fine-tuning completed. Model saved to:", output_dir)


def collate_fn(batch):
    """
    Collate function for padding sequences in the DataLoader.

    Args:
        batch (list[dict]): List of dictionaries containing "input_ids" and "attention_mask".

    Returns:
        dict: Batched input_ids and attention_mask tensors.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":
    # Paths to preprocessed datasets
    train_path = "preprocessed/qa_train_gpt2.parquet"
    dev_path = "preprocessed/qa_dev_gpt2.parquet"

    # Fine-tune GPT-2
    fine_tune_gpt2(train_path, dev_path, output_dir="fine_tuned_models", batch_size=8, epochs=10, lr=10e-5)
