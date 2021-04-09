from utils import *
import argparse

def decompose(data, n_components, method):
    if method == "nmf":
        model = decomposition.NMF(n_components=n_components)
    else:
        model = decomposition.TruncatedSVD(n_components=n_components)
    
    return model.fit_transform(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--method",
            type=str
            )
    parser.add_argument(
            "--f_name",
            type=str
            )

    args = parser.parse_args()
    method = args.method 
    f_name = args.f_name

    data = pd.read_csv(f'../data/{f_name}.csv')

    enc = preprocessing.LabelEncoder()
    y = data["category"].values
    kfold = data['kfold'].values

    X = data.drop(['category', 'kfold'], axis=1)
    df = decompose(data=X, n_components=7000, method=method)

    df = pd.DataFrame(df)
    df["kfold"] = kfold 
    df["category"] = y

    df.to_csv(f'../data/{f_name}_{method}.csv', index=False)