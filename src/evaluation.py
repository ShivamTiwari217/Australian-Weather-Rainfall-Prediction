from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Report": classification_report(y_test, y_pred),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred)
    }
    return metrics
