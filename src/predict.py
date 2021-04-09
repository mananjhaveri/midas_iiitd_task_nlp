from utils import *
import clean 

text = input("Enter description:\n")
text = clean.pipe.transform(text)

tvf = joblib.load('../models/tfidf_10000.bin')
text = tvf.transform(text)

model_tag = input("Enter model tag:\n")

probabilities = None
for fold in range(5):       
    model = joblib.load(f'../models/{model_tag}_{str(fold)}.bin')
    probs =  np.array(model.predict_proba(text)[0])
    print(probs)

probabilities = list(probabilities)
class_ = probabilities.index(max(probabilities)) + 1

enc = joblib.load(f'../models/label_encoder.bin')
class_ = enc.inverse_transform([class_])

for i in range(len(probabilities)):
    print(enc.inverse_transform([i+1]), "-", probabilities[i])

print(class_[0])

