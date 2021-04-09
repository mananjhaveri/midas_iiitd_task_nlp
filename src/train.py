from utils import * 
import argparse

def run(df, fold, ml_model):

    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    x_train = train.drop(["kfold", "category"], axis=1)
    y_train = train.category.values

    x_valid = test.drop(["kfold", "category"], axis=1)
    y_valid = test.category.values

    model = models[ml_model]
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)

    score = metrics.f1_score(y_valid, preds, average="micro")
    conf_mat = metrics.confusion_matrix(y_valid, preds)
    print(f'Fold={fold} Score={score}')
    # joblib.dump(model, f"../models/{ml_model}_{fold}.bin")

    # plt.figure(figsize = (20, 18))
    # conf_mat_plot = sns.heatmap(conf_mat, annot=True, fmt=".2%")
    # plt.show()
    # conf_mat_plot.figure.savefig(f"../media/{ml_model}_{fold}.jpg")

    # d = {
    #     "valid": pd.Series(y_valid).value_counts(),
    #     "preds": pd.Series(preds).value_counts()
    # }
    
    # df = pd.DataFrame(d, index=pd.Series(y_valid).value_counts().index)
    # print(df, "\n\n")

    print(metrics.classification_report(y_valid, preds))

    return score

if __name__ == "__main__":

    # use parser to input which model to use. Model dict present in utils.py
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model",
            type=str
            )
    args = parser.parse_args()
    ml_model = args.model

    # import data
    df = pd.read_csv('../data/tfidf_10000_new.csv')
    print(df.shape)

    y = df["category"].values
    kfold = df['kfold'].values

    X = df.drop(['category', 'kfold'], axis=1)

    # scaling based on model
    if ml_model == "lr" or ml_model == "svc":
        scaler = preprocessing.StandardScaler()
    elif ml_model == "nb":
        scaler = preprocessing.Normalizer()
    try:
        X = scaler.fit_transform(X)
    except:
        pass

    df = pd.DataFrame(X)
    df['kfold'] = kfold 
    df['category'] = y

    # run for each fold, print average of all folds
    average_score = 0
    for fold in range(5):
        average_score += run(df, fold, ml_model)
    
    print(average_score / 5)
