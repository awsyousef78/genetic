import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import time
import os
# created by enas_129380 
def run_ga(file_path):
    # قراءة البيانات
    data = pd.read_csv(file_path)
    target_column = data.columns[-1]

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # تحويل الأعمدة النصية إلى رقمية
    X = pd.get_dummies(X)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]

    # ==================================
    # قبل GA: نموذج كامل
    model_full = RandomForestClassifier()
    start_time = time.time()
    model_full.fit(X_train, y_train)
    y_pred_full = model_full.predict(X_test)
    end_time = time.time()
    acc_full = accuracy_score(y_test, y_pred_full)
    time_full = end_time - start_time

    # رسم Confusion Matrix قبل GA
    cm_full = confusion_matrix(y_test, y_pred_full)
    plt.figure()
    plt.imshow(cm_full, cmap='Blues')
    plt.title("Confusion Matrix Before GA")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm_full.shape[0]):
        for j in range(cm_full.shape[1]):
            plt.text(j, i, cm_full[i, j], ha='center', va='center', color='red')
    cm_full_path = "static/cm_before_ga.png"
    plt.savefig(cm_full_path)
    plt.close()

    # رسم ROC Curve قبل GA
    fpr, tpr, _ = roc_curve(y_test, model_full.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Before GA')
    plt.legend(loc='lower right')
    roc_full_path = "static/roc_before_ga.png"
    plt.savefig(roc_full_path)
    plt.close()

    # ==================================
    # هنا يمكن وضع GA كما في كودك السابق
    # لتقليل الميزات واختيار أفضل مجموعة
    # سأضع مثال مختصر لاختيار كل الأعمدة كمثال
    selected_features = list(range(num_features))  # استبدل هذه بالنتائج النهائية من GA

    model_ga = RandomForestClassifier()
    start_time = time.time()
    model_ga.fit(X_train.iloc[:, selected_features], y_train)
    y_pred_ga = model_ga.predict(X_test.iloc[:, selected_features])
    end_time = time.time()
    acc_ga = accuracy_score(y_test, y_pred_ga)
    time_ga = end_time - start_time

    # Confusion Matrix بعد GA
    cm_ga = confusion_matrix(y_test, y_pred_ga)
    plt.figure()
    plt.imshow(cm_ga, cmap='Blues')
    plt.title("Confusion Matrix After GA")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm_ga.shape[0]):
        for j in range(cm_ga.shape[1]):
            plt.text(j, i, cm_ga[i, j], ha='center', va='center', color='red')
    cm_ga_path = "static/cm_after_ga.png"
    plt.savefig(cm_ga_path)
    plt.close()

    # ROC بعد GA
    fpr, tpr, _ = roc_curve(y_test, model_ga.predict_proba(X_test.iloc[:,1])[:,0] if y.nunique()>2 else model_ga.predict_proba(X_test)[:,1])
    roc_auc_ga = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc_ga:.2f})')
    plt.plot([0,1],[0,1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve After GA')
    plt.legend(loc='lower right')
    roc_ga_path = "static/roc_after_ga.png"
    plt.savefig(roc_ga_path)
    plt.close()

    # حفظ نتائج وهمية للعرض
    results = {
        "accuracy_full": round(acc_full,4),
        "time_full": round(time_full,4),
        "accuracy_ga": round(acc_ga,4),
        "num_features": len(selected_features),
        "best_features": list(X.columns[selected_features]),
        "cm_before": cm_full_path,
        "cm_after": cm_ga_path,
        "roc_before": roc_full_path,
        "roc_after": roc_ga_path
    }

    return results
