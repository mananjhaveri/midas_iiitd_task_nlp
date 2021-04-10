from imblearn import over_sampling, under_sampling, pipeline
from utils import *

if __name__ == "__main__":
    # import data
    df = pd.read_csv('../data/tfidf_25.csv')

    # split X and y
    X = df.drop(['category'], axis=1)
    y = df['category']

    # select only categories with less than 500 samples for upsampling
    d = dict(y.value_counts())
    strategy = dict()
    for k, v in d.items():
        if v > 500:
            strategy[k] = v
        else:
            strategy[k] = 500

    # use SMOTE for upsampling
    over = over_sampling.SMOTE(sampling_strategy=strategy)
    under = under_sampling.RandomUnderSampler(sampling_strategy=strategy)

    steps = [# ('under', under), 
            ('over', over)]

    pipe = pipeline.Pipeline(steps=steps)
    df, y_sm = pipe.fit_resample(X, y)

    print(y_sm.value_counts())

    # save the file back
    df["category"] = y_sm 
    df.to_csv(f'../data/tfidf_resampled.csv', index=False)