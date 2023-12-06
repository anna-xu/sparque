import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def shuffle_split(X,test_size, n_splits=100):
    splits = ShuffleSplit(n_splits, random_state=0, test_size=test_size, train_size=None)
    all_train_inds = []
    all_test_inds = []

    for train_inds, test_inds in splits.split(X):
        all_train_inds += [train_inds]
        all_test_inds += [test_inds]

    return all_train_inds, all_test_inds

def svc_hyperparamterize(X, y, train_inds, Cs=[1e-3,1e-3,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]):
    X_valid_set = X[train_inds]
    y_valid_set = y[train_inds]

    scores = []
    max_score_C = C[0]

    kf = KFold(n_splits = 5, shuffle = True, random_state = 25)
    kf.get_n_splits(X_valid_set)

    C_fold_scores = {'C': [], 'fold': [], 'score': []}

    for C in Cs:
        fold_scores = []
        for i, (train, test) in enumerate(kf.split(X_valid_set)):
            X_train, X_test, y_train, y_test = X_valid_set[train], X_valid_set[test], y_valid_set[train], y_valid_set[test]
            SVC_model = make_pipeline(StandardScaler(), 
                                    SVC(kernel = 'rbf', gamma='scale', C=C)
                                    )
            SVC_model.fit(X_train, y_train)

            score = SVC_model.score(X_test, y_test)

            fold_scores += [score]

            C_fold_scores['fold'] += [i]
            C_fold_scores['C'] += [C]
            C_fold_scores['score'] += [score]

        C_score = np.mean(fold_scores)
    
        if C_score > np.max(scores):
            max_score_C = C
        
    return max_score_C, C_fold_scores

def svc_test(X, y, train_inds, test_inds, C):
    X_train, Y_train = X[train_inds], y[train_inds]
    X_test, Y_test = X[test_inds], y[test_inds]

    SVC_model = make_pipeline(StandardScaler(), 
                              SVC(kernel = 'rbf', gamma='scale', C=C)
                             )
    SVC_model.fit(X_train, Y_train)

    train_pred = SVC_model.predict(X_train)
    test_pred = SVC_model.predict(X_test)

    metric_results = {'accuracy': accuracy_score(Y_train, train_pred),
                      'balanced_accuracy': balanced_accuracy_score(Y_test, test_pred),
                      'precision': precision_score(Y_test, test_pred),
                      'recall': recall_score(Y_test, test_pred),
                      'AUC': roc_auc_score(Y_test, test_pred),
                      'F1': f1_score(Y_test, test_pred)
                     }

    metric_results_df = pd.DataFrame.from_dict(metric_results)

    return metric_results_df

def run_svc_with_shuffle_split(conn_matrix_df):
    if not isinstance(conn_matrix_df, pd.DataFrame):
        conn_matrix_df = pd.read_csv(conn_matrix_df, sep = ',')

    conn_matrix_df['label'] = conn_matrix_df['label'].astype("category")

    nan_df = conn_matrix_df.isna().any(axis=1)
    with open(f'svc_log_{datetime.now()}.txt', 'w') as f:
        f.write(f'rows dropped \n {conn_matrix_df[nan_df]}')

    conn_matrix_df = conn_matrix_df.dropna(axis = 'rows')

    conn_matrix_start = 3
    conn_matrix_end = -2 # since labels is the last column now
    X = conn_matrix_df.iloc[:,conn_matrix_start:conn_matrix_end].copy().values
    y = conn_matrix_df['label'].copy().values

    split_performance = {'split': [], 'C': [], 'accuracy': [], 'balanced_accuracy': [], 'precision': [], 'recall': [], 'AUC': [], 'F1': []}

    train_inds, test_inds = shuffle_split(X, test_size = 0.25, n_splits = 100)
    for i, (train_inds, test_inds) in enumerate(zip(train_inds, test_inds)):
        C, C_fold_scores = svc_hyperparamterize(X, y, train_inds)
        
        metrics_results = svc_test(X, y, train_inds, test_inds, C)

        split_performance['accuracy'] += [metrics_results['accuracy']]
        split_performance['balanced_accuracy'] += [metrics_results['balanced_accuracy']]
        split_performance['precision'] += [metrics_results['precision']]
        split_performance['recall'] += [metrics_results['recall']]
        split_performance['AUC'] += [metrics_results['AUC']]
        split_performance['F1'] += [metrics_results['F1']]
        split_performance['split'] += [i]
        split_performance['C'] += [C]

    split_performance_df = pd.DataFrame.from_dict(split_performance)
    
    return split_performance_df, C_fold_scores