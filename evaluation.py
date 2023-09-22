import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def visualize_cm(y_test:pd.Series, pred:pd.Series) -> None:
    """Visualize confusion matrix of model's predictions"""
    cm = confusion_matrix(y_test, pred, labels=(0, 1), normalize='all')
    cm_df = pd.DataFrame(cm, 
                        columns=["Predicted white wine", "Predicted red wine"], 
                        index=["Actually white wine", "Actually red wine"])
    fig = px.imshow(cm_df, color_continuous_scale='YlOrRd', zmin=0, zmax=1)
    fig.update_layout(font=dict(size=30))
    fig.show()

def performance_metrics(y_test:pd.Series, pred:pd.Series) -> dict:
    """Calculates performance metrics: accuracy, precision, f1 and ROC AUC score. Returns a dictionary with
    the results"""
    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred)
    
    # Calculate precision
    precision = precision_score(y_test, pred)
    
    # Calculate recall 
    recall = recall_score(y_test, pred)
    
    # Calculate F1 score
    f1 = f1_score(y_test, pred)
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, pred)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
