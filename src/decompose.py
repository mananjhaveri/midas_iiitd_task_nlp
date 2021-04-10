from utils import *
import argparse

# function that returns decomposed matrix using method of choice
def decompose(data, n_components, method):
    if method == "nmf":
        model = decomposition.NMF(n_components=n_components)
    else:
        model = decomposition.TruncatedSVD(n_components=n_components)
    
    return model.fit_transform(data)

if __name__ == "__main__":
    # take method and file name as argument
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

    # import data
    data = pd.read_csv(f'../data/{f_name}.csv')

    y = data["category"].values
    kfold = data['kfold'].values

    # calling the decomposition functions
    X = data.drop(['category', 'kfold'], axis=1)
    df = decompose(data=X, n_components=7000, method=method)

    df = pd.DataFrame(df)
    df["kfold"] = kfold 
    df["category"] = y

    df.to_csv(f'../data/{f_name}_{method}.csv', index=False)