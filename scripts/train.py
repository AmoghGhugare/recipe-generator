import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch

print("Loading dataset...")

df = pd.read_csv("RecipeNLG_train_ready_60k.csv")

df = df.dropna()

print("Dataset size:", len(df))

# -----------------------------
# Train Test Split
# -----------------------------

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# -----------------------------
# Load Tokenizer
# -----------------------------

print("Loading tokenizer...")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# -----------------------------
# Preprocessing Function
# -----------------------------

def preprocess(example):

    input_text = "ingredients: " + example["input"] + " recipe:"

    model_inputs = tokenizer(
        input_text,
        max_length=64,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        example["output"],
        max_length=256,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

print("Tokenizing dataset...")

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
val_dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")

model = T5ForConditionalGeneration.from_pretrained("t5-small")

# -----------------------------
# Training Settings
# -----------------------------

training_args = TrainingArguments(

    output_dir="./recipe_model",

    learning_rate=3e-4,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=3,

    logging_steps=100,

    save_strategy="epoch",

    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset
)

# -----------------------------
# Start Training
# -----------------------------

print("Starting training...")

trainer.train()

# -----------------------------
# Save Model
# -----------------------------

print("Saving model...")

trainer.save_model("recipe_generator_model")
tokenizer.save_pretrained("recipe_generator_model")

print("Training complete!")