from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    return rf

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    return lr
