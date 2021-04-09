from utils import * 
from nltk.tokenize import word_tokenize
from sklearn import decomposition 
from sklearn.feature_extraction.text import TfidfVectorizer      

def remove_tabs(s):
    return re.sub('\s+',' ', s).strip()

STOP_WORDS = ["service", "ship", "flipkart", "product", "geniune", "delivery", "online", "best", "price", "discount", "free"
             "key", "feature", "guarantee", "low", "buy", "day", "flipkartcom", "shop", "rs", "brand", "india", "branded", "cash",
             "package", "sale", "days"]

if __name__ == "__main__":

    MAX_FEATURES = 10000
    
    df = pd.read_csv(f'../data/text.csv')
    df["description"] = df["description"].apply(lambda x: remove_tabs(x))

    tvf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, ngram_range=(1, 2), max_features=MAX_FEATURES, stop_words=STOP_WORDS)
    tvf.fit(df["description"])
    text_transformed = tvf.transform(df["description"]).toarray()

    tfidf_df = pd.DataFrame(text_transformed)
    tfidf_df["category"] = df["category"].copy()

    try:
        tfidf_df["kfold"] = df["kfold"].copy()
    except:
        pass

    tfidf_df.to_csv(f'../data/tfidf_{MAX_FEATURES}_new.csv', index=False)
    joblib.dump(tvf, f"../models/tfidf_{MAX_FEATURES}_new.bin")
