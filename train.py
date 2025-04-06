import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# Load the dataset (update filename if different)
df = pd.read_csv("cleaned_dataset.csv")


# ✅ Rename Columns
if "cleaned_tweet" in df.columns:
    df.rename(columns={"cleaned_tweet": "text", "label": "label"}, inplace=True)

# ✅ Remove Missing Values
df.dropna(subset=["text", "label"], inplace=True)

# ✅ Convert Labels ("TRUE"/"FALSE") to Integers (1/0)
df["label"] = df["label"].astype(bool).astype(int)

# ✅ Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# ✅ Load Tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ✅ Tokenization Function
def tokenize_data(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# ✅ Apply Tokenization
dataset = dataset.map(tokenize_data, batched=True)

# ✅ Split Dataset Correctly
dataset = dataset.train_test_split(test_size=0.2)

# ✅ Extract Train & Test Sets
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ✅ Load Pretrained Model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # ✅ Corrected
    eval_dataset=test_dataset,    # ✅ Corrected
    tokenizer=tokenizer,
)

# ✅ Train Model
trainer.train()

# ✅ Save Model
model.save_pretrained("./abuse_detection_model")
tokenizer.save_pretrained("./abuse_detection_model")

print("🚀 Model training completed successfully!")
