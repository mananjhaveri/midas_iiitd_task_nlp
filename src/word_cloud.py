from utils import *
import wordcloud
from PIL import Image

def create_word_cloud(category, text):
   cloud = wordcloud.WordCloud(background_color = "white", max_words = 200)
   cloud.generate(text)
   cloud.to_file(f"../media/{category}.jpg")

if __name__ == "__main__":
    # import data
    df = pd.read_csv("../data/text.csv")

    # load the label encoder for inverse transform of categories
    enc = joblib.load("../models/label_encoder.bin")

    # a dict to map categories to a master string with all descriptions
    text_cats = {}
    for i in range(len(df)):
        cat = str(enc.inverse_transform([df["category"].iloc[i]]).squeeze())
        text_cats[cat] = text_cats.get(cat, "") + " " + df["description"].iloc[i]

    # create word clouds
    for k, v in text_cats.items():
        text = v.strip()
        create_word_cloud(k, text)