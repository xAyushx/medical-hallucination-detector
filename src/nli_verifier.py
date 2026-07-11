from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"

print("nli loading")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

print("nli loaded.")

def verify_claim(claim,evidence_list):
    results=[]
    labels = model.config.id2label
    for i in evidence_list:
        evidence=i["text"]
        
        inputs=tokenizer(evidence,claim,return_tensors="pt")
        with torch.no_grad():
            logits=model(**inputs).logits
        probs=torch.softmax(logits,dim=1)[0]
        
        results.append({
        "pmid": i["pmid"],
    "distance": i["distance"],
    "text": evidence,
    "probabilities": {
        labels[j]: probs[j].item()
        for j in range(len(labels))
    }}) 
    print(model.config.id2label)
    return results