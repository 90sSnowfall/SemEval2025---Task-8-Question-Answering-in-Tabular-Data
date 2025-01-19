import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define hyperparameters for prediction
PREDICT_PARAMS = {
    "max_length": 50,          # Maximum length of generated text
    "temperature": 0.7,        # Sampling temperature (lower = more deterministic)
    "top_p": 0.9,              # Top-p (nucleus) sampling
    "do_sample": True,         # Enable sampling for diverse outputs
    "num_return_sequences": 1, # Number of sequences to generate
    "eos_token_id": 50256,     # End-of-sequence token ID
    "pad_token_id": 50256      # Padding token ID
}

def load_model(model_dir="fine_tuned_models"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict(question, model, tokenizer, device, predict_params):
    input_text = f"<|startoftext|> Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the output using the configured parameters
    output = model.generate(
        input_ids=input_ids,
        **predict_params
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the answer portion
    answer = generated_text.replace(input_text, "").strip()
    return answer


def main():
    print("=== GPT-2 Inference Engine ===")
    print("Type 'exit' to quit.\n")

    model_dir = "fine_tuned_models/epoch_10"  # Adjust the path to your model directory
    model, tokenizer, device = load_model(model_dir)

    while True:
        question = input("Enter your question: ")
        if question.lower() == "exit":
            break

        # Predict the answer
        answer = predict(question, model, tokenizer, device, PREDICT_PARAMS)
        print(f"Generated Answer: {answer}")
        print("-" * 30)


if __name__ == "__main__":
    main()
