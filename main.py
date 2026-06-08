import os, json
import pandas as pd
from collections import Counter
import pickle
import string


BASE_PATH = "."

def load_captions(filepath):
    df = pd.read_csv(filepath)
    captions = {}
    for _, row in df.iterrows():
        img_id = row['image'].split('.')[0]
        captions.setdefault(img_id, []).append(row['caption'])
    return captions

captions_dict = load_captions(os.path.join(BASE_PATH, "captions.txt"))
print(f"pictures count: {len(captions_dict)}")


def clean_caption(caption):
    caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(w for w in caption.split() if w.isalpha() and len(w) > 1)

cleaned_captions = {img_id: [clean_caption(c) for c in caps]
                    for img_id, caps in captions_dict.items()}

with open("cleaned_captions.pkl", "wb") as f:
    pickle.dump(cleaned_captions, f)

with open("cleaned_captions.pkl", "rb") as f:
    cleaned_captions = pickle.load(f)

word_freq = Counter(w for caps in cleaned_captions.values()for cap in caps for w in cap.split())
vocab = {w for w, c in word_freq.items() if c >= 10}
