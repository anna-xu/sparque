import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from datetime import datetime

def svc_with_shuffle_split(X, y):
    splits = ShuffleSplit(n_splits=100, random_state=0, test_size=0.25, train_size=None)

    scores = []

    i = 0

    for train_inds, test_inds in splits.split(X):
        print(f'Currently training on split {i}')

        # training splits
        Xtrain = X[train_inds]
        ytrain = y[train_inds]
        Xtest = X[test_inds]
        ytest = y[test_inds]

        # SVC with training set
        SVC_model = make_pipeline(StandardScaler(), 
                                  SVC(kernel = 'rbf', gamma='scale')
                                 )
        SVC_model.fit(Xtrain, ytrain)

        # test set
        score = SVC_model.score(Xtest, ytest)
        print(score, scores)

        scores += [score]

        i += 1

    return(np.mean(scores), scores)

def run_svc_with_shuffle_split(conn_matrix_df):
    if not isinstance(conn_matrix_df, pd.DataFrame):
        conn_matrix_df = pd.read_csv(conn_matrix_df, sep = ',')

    # if type(conn_matrix_df) == str:
    #     conn_matrix_df = pd.read_csv(conn_matrix_df, sep = ',')

    conn_matrix_df['label'] = conn_matrix_df['label'].astype("category")

    nan_df = conn_matrix_df.isna().any(axis=1)
    with open(f'svc_log_{datetime.now()}.txt', 'w') as f:
        f.write(f'rows dropped \n {conn_matrix_df[nan_df]}')

    conn_matrix_df = conn_matrix_df.dropna(axis = 'rows')

    # X = conn_matrix_df['connectivity'].copy().values
    conn_matrix_start = 3
    conn_matrix_end = -2 # since labels is the last column now
    # X = conn_matrix_df.iloc[:,3:-2].copy().values
    X = conn_matrix_df.iloc[:,conn_matrix_start:conn_matrix_end].copy().values
    y = conn_matrix_df['label'].copy().values

    mean_score, scores = svc_with_shuffle_split(X, y)
    
    return mean_score, scores