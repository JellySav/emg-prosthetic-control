import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    print("\nReporte de Clasificación:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_targets, all_preds)
    return cm