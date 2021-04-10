from utils import *
import string
import textblob
import re
from nltk.stem import PorterStemmer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from nltk.tokenize import MWETokenizer, word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# dictionary for mapping contractions to expanded terms
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

# regex statement to expand contractions
c_re = re.compile('(%s)' % '|'.join(cList.keys()))

lemmatizer = WordNetLemmatizer()

# mapping with pos tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# expanfing contractions
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text.lower())


# tokenization and removing punctuation
def tokenize_and_remove_punct(text):
  text = text.translate(str.maketrans('', '', string.punctuation))
  mtokenizer = MWETokenizer()
  mwe = mtokenizer.tokenize(text.split())
  words =[]
  for t in mwe:
    if t.isalpha():
      words.append(t)
  return words


# adding pos tags
def tags(tokens):
  tags = nltk.pos_tag(tokens)
  return tags
  
# lemmatization
def lemmatize(tags): 
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), tags)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

# removing stop words
def stopword_removal(words):
  stopwords = nltk.corpus.stopwords.words('english')
  newStopWords = ['said','say', 'says','mr']
  stopwords.extend(newStopWords)
  word_filtered = []
  for w in words:
    if w not in stopwords:
        word_filtered.append(w)
  unique = list(dict.fromkeys(word_filtered))
  return " ".join(unique)

# creating pipeline
def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            # return list_or_series
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})


estimators = [("expanding_contractions",pipelinize(expandContractions)),("tokenizer", pipelinize(tokenize_and_remove_punct)),("pos", pipelinize(tags)),("lammatizing", pipelinize(lemmatize)),("removing_stopwords",pipelinize(stopword_removal))]
pipe = Pipeline(estimators)

untidy_categories = []
d = {"Sunglasses": "Eyewear"}# "Food & Nutrition": "Kitchen & Dining", "Household Supplies": "Home Improvement"}

# dictionary to map key words to category for manual labelling
mapping_untidy_categores = {
    "Pens & Stationery": ["paper", "sheets", "self stick", "notebook", "pen"],
    "Clothing": ["kurta", "kurti","stole", "dress", "shorts", "capri", "jeans", "top", "brief", "jumpsuit", "sleeve", "boxer", "leggings", "shirt",
                "socks", "round neck", "bra", "polo neck", "jacket", "vest", "trouser", "pyjama", "camisole", "lingerie", "blazer", "bottom",
                "sari", "cotton", "cufflink", "bangle", "hair clip", "hair band", "glove", "strap"],
    "Home Furnishing": ["sofa cover", "mattress", "mat", "king size", "tapestry", "furnishing", 
                        "polyester", "single bed", "faucet"],
    "music": ["sound mixer", "headset"],
    "footwear": ["shoes", "flats", "wedges", "boot", "bellies", "foot", "toe", "lace up", "slippers", "abeez"],
    "Bags, Wallets & Belts": ["clutch", "bag", "bug"],
    "Eyewear": ["sunglasses"],
    "Kitchen & Dining": ["table cover", "glass", "bowl", "ro water"],
    "Sports & Fitness": ["thigh guard"],
    "Beauty and Personal Care": ["nail", "nail cutter", "shampoo", "conditioner", "conditione", "brush",
                              "shaving", "shaving kit", "hair"],
    "Health and Personal Care Appliances": ["contamination", "wash your hands"],
    "Jewellery": ["ring", "swarovski", "royal seal creation"], 
    "Toys & School Supplies": ["game"], 
    "Automotive": ["car", "shade", "bike", "speed", "wheel", "1 compartments", "vehicle"],
    "Mobiles & Accessories": ["battery", "acer", "0mah", "tablet", "charging", "blackberry"],
    "Home Decor & Festive Needs": ["plant", "candle", "showpiece", "incense"],
    "Automation & Robotics": ["lock", "safe"],
    "Home & Kitchen": ["bottle"],
    "Food & Nutrition": ["seed", "herb"],
    "Tools & Hardware": ["mcb", "workbench", "motion sensor", "pump", "usb", "binoculars", "work bench", "bino"],
    "Baby Care": ["baby"]
}

# clean labels, extract parent category or map using above dictionary
def clean_labels(x):
    if ">>" in x:
        y = x.split(">>")[0].strip()[2:]
        try:
            return d[y]
        except:
            return y
    else:
        y = x.lower()
        for k, v in mapping_untidy_categores.items():
            for cats in v:
                if cats in y:
                    return k
        untidy_categories.append(x)
        return "NONE"

# threshold setting for counts of classes
def min_samples(df, n_min_samples):
    ret = []
    count = df.category.value_counts()
    for cat in df.category.unique():
        if count[cat] > n_min_samples:
            ret.append(cat)
    return ret

# plotting bar graph of categories and count 
def get_bar(x, y):
    fig = plt.figure(figsize = (12, 10))
    plt.bar(x, y, color ='maroon',
            width = 0.4)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation='vertical')
    plt.xlabel("categories")
    plt.ylabel("count")
    plt.title("category and counts")
    plt.show()


if __name__ == "__main__":
    # import data
    df = pd.read_csv("../data/data.csv")

    df = df[["description", "product_category_tree", "image"]].copy()
    df = df.dropna()

    # cleaning targets
    df["category"] = df["product_category_tree"].apply(clean_labels)
    df.drop(["product_category_tree"], axis=1, inplace=True)
    vc = df.category.value_counts()
    # get_bar(list(vc.index), list(vc))

    # cleaning description
    cleaned_text = []
    for t in df['description']:
        cleaned_text.append(pipe.transform([t])[0])
    df['description'] = cleaned_text

    # setting threshold of 25 for counts of classes
    for N_MIN_SAMPLES in [25]:
        final_text = df[df['category'].map(df['category'].value_counts()) >= N_MIN_SAMPLES].copy()
        vc = final_text.category.value_counts()
        # get_bar(list(vc.index), list(vc))

        final_text = create_folds(final_text)

        enc = preprocessing.LabelEncoder()
        final_text["category"] = enc.fit_transform(final_text["category"])

        joblib.dump(enc, "../models/label_encoder.bin")
        final_text.to_csv(f"../data/text.csv", index=False)
