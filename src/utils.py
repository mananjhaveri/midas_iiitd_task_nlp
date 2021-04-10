import pandas as pd 
import numpy as np 
from sklearn import model_selection, preprocessing, metrics
from sklearn import naive_bayes, ensemble, linear_model, svm, decomposition, tree
import re 
import io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost 

# use the following models:
models = {
    "nb": naive_bayes.GaussianNB(),
    "rf": ensemble.RandomForestClassifier(n_estimators=500),
    "lr": linear_model.LogisticRegression(multi_class='multinomial'),
    "svc": svm.SVC(),
    "dt": tree.DecisionTreeClassifier(),
    "xgb": xgboost.XGBClassifier(n_jobs=-1)
}

# create folds for cross validation 
def create_folds(data):
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.category.values)):
        data.loc[v_, 'kfold'] = f

    return data
