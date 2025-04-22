from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load saved model and tokenizer
model_path = "./abuse_detection_model"  # Update path if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()
def predict_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():  # No gradient computation (faster)
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()  # Get predicted class (0 or 1)
    
    return "Non-Abusive ✅" if predicted_label == 1 else "Abusive ❌"
test_comments = [
    "You , look like a nigga",
    " wtf",
    "That's a wonderful idea!",
    "good feel!"
]

for comment in test_comments:
    print(f"Comment: {comment} → Prediction: {predict_comment(comment)}")
