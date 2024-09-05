from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
import numpy as np

def comp_auc(y_label,y_predict):
    fpr, tpr, threshods = metrics.roc_curve(np.array(y_label), np.array(y_predict), pos_label=1)

    return metrics.auc(fpr,tpr)

def comp_f1score(y_label,y_predict):

    return f1_score(y_label,y_predict)


def classi_report(y_label,y_predict):
    return classification_report(y_label,y_predict,labels=[0,1],output_dict=True)


def comp_sensitivity(true_labels,pred_labels):
#     fpr, tpr, threshods = metrics.roc_curve(np.array(y_label), np.array(y_predict), pos_label=0)
    
#     return recall_score(y_label,y_predict,pos_label=0,average='weighted')
    # return recall_score(y_label,y_predict,pos_label=0,average='weighted')
  
    true_positive = sum([1 for pred, true in zip(pred_labels, true_labels) if pred == True and true == True])
    # 计算所有实际负类样本的数量
    actual_positive = sum([1 for label in true_labels if label == True])
    if actual_positive == 0:
        sensitivity = 0
    else:
        #    计算特异度
        sensitivity = true_positive / actual_positive#actual_neg可能为0
    
    return sensitivity

def comp_specificity(true_labels,pred_labels):
#     fpr, tpr, threshods = metrics.roc_curve(np.array(y_label), np.array(y_predict), pos_label=0)
    
#     return recall_score(y_label,y_predict,pos_label=0,average='weighted')
    # return recall_score(y_label,y_predict,pos_label=0,average='weighted')
  
    true_negatives = sum([1 for pred, true in zip(pred_labels, true_labels) if pred == False and true == False])
    # 计算所有实际负类样本的数量
    actual_negatives = sum([1 for label in true_labels if label == False])
    if actual_negatives == 0:
        specificity = 0
    else:
        #    计算特异度
        specificity = true_negatives / actual_negatives#actual_neg可能为0
    
    return specificity


