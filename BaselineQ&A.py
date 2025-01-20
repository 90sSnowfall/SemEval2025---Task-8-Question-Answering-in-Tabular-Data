!pip install transformers datasets
from google.colab import drive

import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
#from transformers import TrainerCallback
from transformers import GPT2Config

# Încarcă datele din fișierul CSV
file_path = "/content/databench_train.csv"  # file_path = "/content/drive/My Drive/databench_train.csv"
df = pd.read_csv(file_path)

# Pre-procesarea datelor
# Selectăm pentru train_df: valorile cu 'dataset' între 001 și 049 și indici între 1 și 989
train_df = df[
    (df['dataset'].str[:3].astype(int).between(1, 49)) &  # Prefixul numeric între 001 și 049
    (df.index >= 2) &
    (df.index <= 22596)
]

# Selectăm pentru dev_df: 'dataset' între 050 și 065 și indicii între 22597 și 30001
dev_df = df[
    (df['dataset'].str[:3].astype(int).between(50, 65)) &  # Prefixul numeric între 050 și 065
    (df.index >= 22597) &
    (df.index <= 30001)
]

# Conversia datelor într-un format compatibil cu dataset-ul Hugging Face
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)

# ----------------------------- Augmentare (opțională) -----------------------------
# Exemplu: Shuffle aleator pentru fraze
# def augment_data(example):
#     import random
#     text = example['question']
#     if random.random() > 0.5:
#         words = text.split()
#         random.shuffle(words)
#         example['question'] = ' '.join(words)
#     return example

# Aplica augmentare pe train_dataset:
# train_dataset = train_dataset.map(augment_data)
# ----------------------------------------------------------------------------------
# Setează parametrii modelului, inclusiv dropout
config = GPT2Config.from_pretrained(
    "gpt2-medium",  # Poți înlocui cu modelul dorit, de exemplu "gpt2" sau "gpt2-large"
    attention_probs_dropout_prob=0.1,  # Dropout pentru atenție
    hidden_dropout_prob=0.1,            # Dropout pentru straturi ascunse
    resid_pdrop=0.1                     # Dropout pentru reziduuri
)
# Încarcă modelul și tokenizer-ul GPT-2
MODEL_NAME = "gpt2-medium"  #  ( "gpt2" dacă resursele sunt limitate)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Setează token-ul de padding
tokenizer.pad_token = tokenizer.eos_token

# Funcție pentru tokenizare
def tokenize_function(examples):
    encodings = tokenizer(examples['question'], truncation=True, padding="longest", max_length=128)
    encodings['labels'] = encodings['input_ids']
    return encodings

# Aplica tokenizarea pe întregul dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)

# Pregătește argumentele pentru antrenament
output_dir = "/content/Model3"
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    logging_dir="/content/logs",
    fp16=True,  # Activează mixed precision
    report_to="none"
    adam_epsilon=1e-8  # Parametru de regularizare pentru optimizator
)

# ----------------------------- Gradual Layer Freezing (GLF) -----------------------------
# Funcție pentru dezactivarea treptată a straturilor inferioare
# def gradual_layer_freezing(model, epoch):
#     for i, layer in enumerate(model.transformer.h):
#         if i < epoch:  # Dezactivează straturile inferioare
#             for param in layer.parameters():
#                 param.requires_grad = False
#         else:  # Straturile superioare rămân antrenabile
#            for param in layer.parameters():
#                param.requires_grad = True

# Callback pentru gradual layer freezing
#class GradualLayerFreezingCallback(TrainerCallback):
#    def __init__(self, model):
#        self.model = model
#
#    def on_epoch_begin(self, args, state, control, **kwargs):
#        gradual_layer_freezing(self.model, state.epoch)
#
#    def on_init_end(self, args, state, control, **kwargs):
#        pass

#Exemplu adaugare la final in Trainer:callbacks=[EarlyStoppingCallback(early_stopping_patience=2), GradualLayerFreezingCallback(model)],

# Exemplu: Adăugare GLF în timpul trainingului
# def glf_callback(trainer, epoch):
#     gradual_layer_freezing(trainer.model, epoch)
# ---------------------------------------------------------------------------------------

# Crează un obiect Trainer pentru a antrena modelul
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping

    # ----------------------------- Metrici Personalizate (opțional) -----------------------------
    # compute_metrics=compute_f1  # Exemplu: metrică F1 Score
    # compute_metrics=compute_bleu  # Exemplu: metrică BLEU Score
    # --------------------------------------------------------------------------------------------
)

# ----------------------------- Metrici: F1 Score și BLEU Score -----------------------------
# Exemplu F1 Score
# def compute_f1(eval_pred):
#     predictions, labels = eval_pred
#     predictions = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
#     labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
#     correct = sum(1 for p, l in zip(predictions, labels) if p == l)
#     precision = correct / len(predictions) if predictions else 0
#     recall = correct / len(labels) if labels else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     return {"f1": f1}

# Exemplu BLEU Score
# from datasets import load_metric
# bleu_metric = load_metric("bleu")

# def compute_bleu(eval_pred):
#     predictions, labels = eval_pred
#     predictions = [tokenizer.decode(p, skip_special_tokens=True).split() for p in predictions]
#     labels = [[tokenizer.decode(l, skip_special_tokens=True).split()] for l in labels]
#     bleu = bleu_metric.compute(predictions=predictions, references=labels)
#     return {"bleu": bleu['bleu']}
# --------------------------------------------------------------------------------------------

# Începem antrenamentul
trainer.train()

# Salvăm modelul după antrenament
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Antrenament complet și modelul a fost salvat!")
