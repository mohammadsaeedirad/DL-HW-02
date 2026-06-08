import os, json
import pandas as pd
from collections import Counter
import pickle
import string
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet50(pretrained=True)
encoder_model = torch.nn.Sequential(*list(base_model.children())[:-1])
encoder_model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def encode_images(img_ids, img_dir, save_path):
    features = {}
    with torch.no_grad():
        for img_id in img_ids:
            img_path = os.path.join(img_dir, img_id + ".jpg")  # ← اصلاح شد
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            feat = encoder_model(transform(img).unsqueeze(0).to(device))
            features[img_id] = feat.squeeze().cpu().numpy()
    with open(save_path, "wb") as f:
        pickle.dump(features, f)
    print(f"encoded {len(features)}/{len(img_ids)} images → {save_path}")
    return features

img_dir = os.path.join(BASE_PATH, "Images")

sample_files = os.listdir(img_dir)[:3]
print("sample files in Images in d dimention:", sample_files)
print("sample train_ids:", train_ids[:3])

train_features = encode_images(train_ids, img_dir, "train_features.pkl")
test_features  = encode_images(test_ids,  img_dir, "test_features.pkl")

if train_features:
    print(f"train features: {len(train_features)}, shape: {next(iter(train_features.values())).shape}")
else:
    print("train_features خالیه! نام فایل‌ها رو بررسی کن.")


special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
full_vocab = special_tokens + sorted(list(vocab))

word2idx = {w: i for i, w in enumerate(full_vocab)}
idx2word = {i: w for w, i in word2idx.items()}

VOCAB_SIZE = len(word2idx)
print(f"vocab final size: {VOCAB_SIZE}")

max_len = max(
    len(cap.split())
    for caps in train_captions.values()
    for cap in caps
)
print(f"caption max lenght: {max_len}")