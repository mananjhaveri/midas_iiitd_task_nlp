from utils import *

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):

    words = str(s).lower()
    words = tokenizer(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]

    M = []
    for w in words:
        if w in embedding_dict:
            M.append(embedding_dict[w])

    if len(M) == 0:
        return np.zeros(300)

    M = np.array(M)
    v = M.sum(axis=0)

    return v / np.sqrt((v ** 2).sum())


def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
            fname,
            'r',
            encoding='utf-8',
            newline='\n',
            errors='ignore'
        )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))

    return data

if __name__ == "__main__":

    df = pd.read_csv('../data/text_25.csv')
    y = df.category.values
    enc = preprocessing.LabelEncoder()
    df["category"] = enc.fit_transform(df["category"])

    embeddings = load_vectors('../data/crawl-300d-2M.vec')

    vectors = []
    for description in df.description.values:
        vectors.append(
                sentence_to_vec(
                s = description,
                embedding_dict = embeddings,
                stop_words = [],
                tokenizer = word_tokenize
                )
            )

    vectors = np.array(vectors)
    df = pd.DataFrame(vectors)
    df["category"] = y

    df = create_folds(df)
    df.to_csv('../data/fast_text.csv', index=False)

    print(df.shape)