import os, json
import pandas as pd
from collections import Counter
import pickle
import string
import random


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
print(f"vocabulary size: {len(vocab)}")


df = pd.read_csv(os.path.join(BASE_PATH, "captions.txt"))
all_ids = [x.split('.')[0] for x in df['image'].unique()]
random.seed(42)
random.shuffle(all_ids)

train_ids = all_ids[:6000]
test_ids  = all_ids[6000:]

with open("Flickr_8k.trainImages.txt", "w") as f:
    f.write("\n".join(train_ids))
with open("Flickr_8k.testImages.txt", "w") as f:
    f.write("\n".join(test_ids))

print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

train_img_ids = set(train_ids)
train_captions = {img_id: [f"<START> {cap} <END>" for cap in caps]
                  for img_id, caps in cleaned_captions.items()
                  if img_id in train_img_ids}

print(f"train pictures count: {len(train_captions)}")