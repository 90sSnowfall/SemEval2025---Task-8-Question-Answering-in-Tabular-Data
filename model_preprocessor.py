import os
import pandas as pd
from transformers import GPT2Tokenizer
from datasets import load_dataset


def save_splits(df_train, df_dev, output_dir="preprocessed"):
    """
    Save train and dev splits as both .parquet and .csv files.

    Parameters:
        df_train (pd.DataFrame): Train dataset.
        df_dev (pd.DataFrame): Dev dataset.
        output_dir (str): Directory to save the files. Defaults to 'preprocessed'.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    print(f"Saving preprocessed splits to: {output_dir}")

    # Save train split
    df_train.to_parquet(os.path.join(output_dir, "qa_train.parquet"), index=False)
    df_train.to_csv(os.path.join(output_dir, "qa_train.csv"), index=False)
    print("Train split saved.")

    # Save dev split
    df_dev.to_parquet(os.path.join(output_dir, "qa_dev.parquet"), index=False)
    df_dev.to_csv(os.path.join(output_dir, "qa_dev.csv"), index=False)
    print("Dev split saved.")

def load_splits(input_dir="preprocessed"):
    """
    Load train and dev splits from .parquet files.

    Parameters:
        input_dir (str): Directory containing the saved splits.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded train and dev DataFrames.
    """
    train_path = os.path.join(input_dir, "qa_train.parquet")
    dev_path = os.path.join(input_dir, "qa_dev.parquet")

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        raise FileNotFoundError(f"Parquet files not found in {input_dir}. Ensure preprocessing is complete.")

    print(f"Loading train split from: {train_path}")
    df_train = pd.read_parquet(train_path)

    print(f"Loading dev split from: {dev_path}")
    df_dev = pd.read_parquet(dev_path)

    return df_train, df_dev

def preprocess_text(text):
    """
    Preprocess a single text field.
    - Lowercases the text
    - Removes unnecessary spaces
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def tokenize_field(field, field_type, tokenizer):
    # Print the input to the function
    print(f"Field: {field}, Field Type: {field_type}")

    # Handle boolean values
    if field_type == "boolean":
        text = str(field).lower()  # Convert True/False to "true"/"false"
        print(f"Processed boolean: {text}")

    # Handle numeric values (int or float)
    elif "number" in field_type:
        text = str(field)
        print(f"Processed number: {text}")

    # Handle lists
    elif "list" in field_type:
        if isinstance(field, list):
            text = " ".join(map(str, field))
            print(f"Processed list: {text}")
        else:
            text = str(field)
            print(f"Non-list field treated as list: {text}")

    # Handle strings
    elif isinstance(field, str):
        text = field
        print(f"Processed string: {text}")

    # Fallback for unexpected or empty fields
    else:
        text = "unknown"
        print(f"Fallback processing: {text}")

    # Tokenize the text
    try:
        tokens = tokenizer(text=text, return_tensors="pt", truncation=True, padding=True)["input_ids"].squeeze().tolist()
        print(f"Tokens: {tokens}")
        return tokens if isinstance(tokens, list) else []
    except Exception as e:
        print(f"Tokenization error for field: {field} (Type: {field_type}): {e}")
        return []


def tokenize_dataset(input_path, output_parquet, output_csv, tokenizer):
    """
    Tokenizes a dataset and saves the results as .parquet and .csv files.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Processing {input_path}...")
    df = pd.read_parquet(input_path)

    # Preprocess text fields
    df["question"] = df["question"].apply(preprocess_text)
    df["answer"] = df["answer"].fillna("")

    # Tokenize fields
    df["question_tokens"] = df["question"].apply(
        lambda x: tokenizer(text=x, return_tensors="pt", truncation=True, padding=True)["input_ids"].squeeze().tolist()
    )
    df["answer_tokens"] = df.apply(
        lambda row: tokenize_field(row["answer"], row["type"], tokenizer), axis=1
    )

    # Ensure all tokenized fields are lists
    df["question_tokens"] = df["question_tokens"].apply(lambda x: x if isinstance(x, list) else [])
    df["answer_tokens"] = df["answer_tokens"].apply(lambda x: x if isinstance(x, list) else [])

    # Save tokenized dataset
    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_csv, index=False)
    print(f"Saved tokenized dataset to {output_parquet} and {output_csv}")


def main():
    input_dir = "preprocessed"
    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading the Databench QA subset...")
    qa_data = load_dataset("cardiffnlp/databench", "qa", split="train").to_pandas()

    df_train = qa_data[qa_data['dataset'].str.startswith(tuple(f"{i:03}" for i in range(1, 50)))]
    df_dev = qa_data[qa_data['dataset'].str.startswith(tuple(f"{i:03}" for i in range(50, 66)))]
    print(f"Train split: {len(df_train)} rows, Dev split: {len(df_dev)} rows.")

    save_splits(df_train, df_dev)

    # Define file paths
    train_input_path = os.path.join(input_dir, "qa_train.parquet")
    dev_input_path = os.path.join(input_dir, "qa_dev.parquet")

    train_output_parquet = os.path.join(output_dir, "qa_train_gpt2.parquet")
    train_output_csv = os.path.join(output_dir, "qa_train_gpt2.csv")

    dev_output_parquet = os.path.join(output_dir, "qa_dev_gpt2.parquet")
    dev_output_csv = os.path.join(output_dir, "qa_dev_gpt2.csv")

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Tokenize train and dev splits
    tokenize_dataset(train_input_path, train_output_parquet, train_output_csv, tokenizer)
    tokenize_dataset(dev_input_path, dev_output_parquet, dev_output_csv, tokenizer)


if __name__ == "__main__":
    main()
