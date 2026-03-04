# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:14:50 2026

@author: User
"""
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score

# Function to print confusion matrix and evaluation matrix
def evaluation(df,y_true,y_pred):
    df = df.dropna(subset=[y_true,y_pred]).copy()

    # Confusion matrix
    cm = confusion_matrix(df[y_true], df[y_pred])
    
    # Compute evaluation matrix
    TN, FP, FN, TP = cm.ravel()
    
    # Print confusion matrix
    print("Model:", y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nMetrics:")
    print("Precision:", precision_score(df[y_true], df[y_pred]))
    print("Recall (Sensitivity):", recall_score(df[y_true], df[y_pred]))
    print("F1 Score:", f1_score(df[y_true], df[y_pred]))
    print("Specificity:", TN / (TN + FP))
    print("===================================")

# Load datasets
predicted_df = pd.read_csv("classified_output.csv")

# Convert dataframe
predicted_df['inappropriate_yesno'] = predicted_df["inappropriate_yesno"].map({
    "Yes": 0,
    "No": 1
})

##### Print all confusion matrices
evaluation(predicted_df,"inappropriate_yesno","google/gemma-3-1b")
evaluation(predicted_df,"inappropriate_yesno","google/gemma-3-12b")
evaluation(predicted_df,"inappropriate_yesno","google/gemma-3-27b")
evaluation(predicted_df,"inappropriate_yesno","openai/gpt-oss-20b")



