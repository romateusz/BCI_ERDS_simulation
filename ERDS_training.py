import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

def podziel_dane_do_uczenia(daneLewa, danePrawa):

    X = np.concatenate([daneLewa, danePrawa], axis=0)
    Y = np.array([0] * 30 + [1] * 30)  # 0 - lewa, 1 - prawa
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)
    
    # Wyświetlenie kształtów
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_train shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def treningRegresjaLogistyczna(X_train, y_train):
    # Train Logistic Regression
    model = LogisticRegression(max_iter=200, C=10.0)
    model.fit(X_train, y_train)
    
    # Predictions with Logistic Regression
    y_train_pred_lr = model.predict(X_train)
    
    # Accuracy for Logistic Regression
    train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
    print("\nLogistic Regression Train Accuracy:", train_accuracy_lr)
    return model

def testRegresjaLogistyczna(model, X_test, y_test):
    # Predictions with Logistic Regression
    y_test_pred_lr = model.predict(X_test)
    test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)

    # Confusion Matrix and Classification Report for Logistic Regression
    conf_matrix_lr = confusion_matrix(y_test, y_test_pred_lr)
    class_report_lr = classification_report(y_test, y_test_pred_lr)

    print("Logistic Regression Test Accuracy:", test_accuracy_lr)
    print("Logistic Regression Confusion Matrix:\n", conf_matrix_lr)
    print("Logistic Regression Classification Report:\n", class_report_lr)


def treningXGBOOST(X_train, y_train):
    
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    model.fit(X_train, y_train)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Najlepsze parametry i wynik
    print("Najlepsze parametry:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions with Logistic Regression
    y_train_pred_xgb = best_model.predict(X_train)
    
    # Accuracy
    train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
    return best_model

def testTESTXGBOOST(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    print(f"Dokładność po tuningu: {accuracy_score(y_test, y_pred):.4f}")
    y_test_pred_xgb = best_model.predict(X_test)
    test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)

    # Confusion Matrix and Classification Report
    conf_matrix_xgb = confusion_matrix(y_test, y_test_pred_xgb)
    class_report_xgb = classification_report(y_test, y_test_pred_xgb)
    print("Test Accuracy:", test_accuracy_xgb)
    print("Confusion Matrix:\n", conf_matrix_xgb)
    print("Classification Report:\n", class_report_xgb)