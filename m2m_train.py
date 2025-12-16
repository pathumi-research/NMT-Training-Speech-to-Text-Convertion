# ====================================================================
# SCRIPT: nmt_m2m100_local.py
# PROJECT: NMT-Training-Speech-to-Text-Convertion (Sinhala Glossing Fine-tuning)
# DESCRIPTION: Fine-tunes the M2M100 multilingual model for Sinhala (si) 
#              to Gloss (si) sequence-to-sequence translation.
# ====================================================================

# ====================================================================
# STEP 0: Setup and Installation
# ====================================================================

import os
import torch
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from torch.optim import AdamW

# Set up CUDA if a GPU is available (highly recommended for Transformers)
# Note: The notebook was run on 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define File Paths (Adjust to local paths if not using Google Colab/Drive)
# Original path: /content/drive/MyDrive/research/model/merged_f.txt
# Assuming the file is now in the same directory as the script or accessible via a local path:
FILE_PATH = "merged_f.txt" # You MUST ensure this file exists locally.
OUTPUT_DIR = "m2m100_finetuned" 

# ====================================================================
# STEP 1: Data Loading, Splitting, and Initial Dataset Creation
# ====================================================================

src_texts = []
tgt_texts = []

print("\n1. Loading and parsing data...")
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "@" in line:
                parts = line.strip().split("@")
                if len(parts) == 2:
                    # Source (Sinhala Text) is before '@'
                    src_texts.append(parts[0].strip())
                    # Target (Gloss IDs/Text) is after '@'
                    tgt_texts.append(parts[1].strip())
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please check your path.")
    exit()

print(f"Total samples loaded: {len(src_texts)}")

# Split data: 90% Train, 10% Eval
train_src, eval_src, train_tgt, eval_tgt = train_test_split(
    src_texts, tgt_texts, test_size=0.1, random_state=42
)

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_dict({"source": train_src, "target": train_tgt})
eval_dataset = Dataset.from_dict({"source": eval_src, "target": eval_tgt})

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")


# ====================================================================
# STEP 2: Model and Tokenizer Initialization (Loading from Local Path)
# ====================================================================

# This is the path where the base model files were saved locally 
# (corresponding to m2m100_base_saved in your original environment)
model_save_path = "m2m100_base_saved" 

MODEL_NAME = "facebook/m2m100_418M" # Original Hugging Face name

print("\n2. Initializing model...")

# Check if the model already exists locally
if os.path.exists(model_save_path):
    print(f"📂 Found saved model locally. Loading from: {model_save_path}")
    # Load directly from the local directory
    model = M2M100ForConditionalGeneration.from_pretrained(model_save_path).to(device)
    tokenizer = M2M100Tokenizer.from_pretrained(model_save_path)

else:
    # If not found locally, download and then save for the future.
    # Note: This is crucial if you are deploying to a new machine.
    print(f"⬇️ Model not found locally. Downloading from Hugging Face: {MODEL_NAME}")
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)

    # Save to the local directory for next time
    print(f"💾 Saving model to local directory for future use: {model_save_path}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

# Set Languages (Source and Target are both 'si' for glossing task)
tokenizer.src_lang = "si"
tokenizer.tgt_lang = "si"
print(f"Tokenizer set for src_lang: {tokenizer.src_lang} and tgt_lang: {tokenizer.tgt_lang}")


# ====================================================================
# STEP 3: Vocabulary Adaptation for Target Gloss Tokens
# ====================================================================

print("\n3. Adapting vocabulary for Gloss tokens...")

# 3.1 Extract unique Gloss Tokens
unique_gloss_tokens = set()
all_tgt_texts = train_tgt + eval_tgt

for text in all_tgt_texts:
    parts = text.split('|')
    for part in parts:
        if ':' in part:
            # Take everything after the first ':' (the Gloss word)
            gloss_token = part.split(':', 1)[1]
            unique_gloss_tokens.add(gloss_token.strip())

unique_gloss_tokens_list = list(unique_gloss_tokens)
print(f"Identified {len(unique_gloss_tokens_list)} unique Gloss tokens.")

# 3.2 Add new tokens and resize model embeddings
num_added_toks = tokenizer.add_tokens(unique_gloss_tokens_list)
print(f"Added {num_added_toks} new tokens to the tokenizer.")

# Resize the model embeddings to include the new tokens.
model.resize_token_embeddings(len(tokenizer))
print(f"Model vocabulary size resized to: {len(tokenizer)}")


# ====================================================================
# STEP 4: Tokenization Function and Dataset Mapping
# ====================================================================

MAX_SEQ_LENGTH = 128

def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]

    # Tokenize inputs (Source - Sinhala Text)
    model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True)

    # Tokenize targets (Labels - Gloss Text)
    # Using text_target is the modern approach for target tokenization
    labels = tokenizer(text_target=targets, max_length=MAX_SEQ_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization to datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Remove original columns
tokenized_train = tokenized_train.remove_columns(["source", "target"])
tokenized_eval = tokenized_eval.remove_columns(["source", "target"])


# ====================================================================
# STEP 5: Differential Learning Rate Configuration
# ====================================================================

print("\n4. Configuring Differential Learning Rates...")

# Differential LRs based on the paper's strategy
LR_NEW_WEIGHTS = 1.0e-3 # For newly initialized/resized embeddings and output head
LR_PRE_TRAINED = 2.0e-5 # For the core pre-trained layers (Encoder/Decoder)

# Separate parameters into two groups
# Group 1: Pre-trained layers (core Encoder/Decoder) -> SMALL LR
pretrained_params = [
    p for n, p in model.named_parameters() if p.requires_grad and not any(ext in n for ext in ["embed", "lm_head"])
]

# Group 2: Newly resized embeddings and output head -> LARGER LR
new_params = [
    p for n, p in model.named_parameters() if p.requires_grad and any(ext in n for ext in ["embed", "lm_head"])
]

# Define the parameter groups for the optimizer
optimizer_grouped_parameters = [
    {
        "params": pretrained_params,
        "lr": LR_PRE_TRAINED,
        "weight_decay": 0.01
    },
    {
        "params": new_params,
        "lr": LR_NEW_WEIGHTS,
        "weight_decay": 0.01
    },
]


# ====================================================================
# STEP 6: Setup Trainer Arguments and Initialization
# ====================================================================

BATCH_SIZE = 4
NUM_EPOCHS = 10
SAVE_STEPS = 500

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8, # Effective batch size = 4 * 8 = 32
    warmup_ratio=5/NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",

    # Checkpointing Configuration
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="epoch",
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    learning_rate=LR_PRE_TRAINED, # Note: This default LR is overridden by the AdamW optimizer config below
    safe_serialization=True
)

# Initialize Trainer. 
# The AdamW optimizer is passed with the differential LRs (optimizer_grouped_parameters)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(AdamW(optimizer_grouped_parameters, eps=1e-6), None),
)


# ====================================================================
# STEP 7: Start Fine-Tuning (with Resume Capability)
# ====================================================================

print("\n5. Starting fine-tuning...")

# Check for the latest checkpoint in the output directory
latest_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    # Find the latest checkpoint folder
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-")
    ]
    if checkpoints:
        # Get the path of the most recently modified checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Found existing checkpoint: {latest_checkpoint}. Resuming training...")

# Start training, resuming if a checkpoint was found
train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)

print("\nFine-tuning complete!")

# Save the final model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Final model and tokenizer saved to: {OUTPUT_DIR}")

"""
pip install \
    transformers==4.30.2 \
    datasets==2.14.0 \
    sentencepiece==0.1.99 \
    accelerate==0.21.0 \
    scikit-learn==1.3.0 \
    torch==2.6.0
    # Install torchmetrics separately or add it to the list above
pip install torchmetrics==1.8.2
"""