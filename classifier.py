# =============================
# 🔹 라이브러리 임포트
# =============================
import os
import re
import json
import pandas as pd
from html import unescape
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

print("classifier.py 시작됨")

# =============================
# 🔹 1. 텍스트 정제 함수 정의
# =============================
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)                    # HTML 태그 제거
    text = unescape(text)                                 # HTML 엔티티 디코딩
    text = re.sub(r'[^\x20-\x7E]', ' ', text)             # 비프린터블 문자 제거
    text = re.sub(r'\s+', ' ', text)                      # 공백 정리
    return text.lower().strip()                           # 소문자화 및 앞뒤 공백 제거

# =============================
# 🔹 2. JSON 파일 로딩 및 텍스트 + 라벨 정리
# =============================
def load_data():
    label_map = {
        "HR.json": 0,  # Human Real
        "HF.json": 1,  # Human Fake
        "MR.json": 2,  # Machine Real
        "MF.json": 3   # Machine Fake
    }

    data = []
    base_path = "C:/CONTEST_SW/dataset"

    for fname, label_id in label_map.items():
        file_path = os.path.join(base_path, fname)
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            for key, sample in samples.items():
                _id = sample.get("id", f"unknown-{fname}-{key}")
                title = sample.get("title", "")
                desc = sample.get("description", "")
                text = sample.get("text", "")
                full_text = f"{_id}. {title} {desc} {text}"
                if full_text.strip():
                    data.append((clean_text(full_text), label_id))
    
    return pd.DataFrame(data, columns=["text", "label"])

def get_dataloaders(batch_size=4):
    df = load_data()
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

    def tokenize(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    train_encodings = tokenize(train_texts)
    val_encodings = tokenize(val_texts)
    test_encodings = tokenize(test_texts)

    class NewsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# =============================
# 🔹 7. 단독 실행 시 확인용 출력
# =============================
if __name__ == "__main__":
    df = load_data()
    print(df.head(3))
    print("전체 샘플 수:", len(df))
    train_loader, val_loader, test_loader = get_dataloaders()
    print("토큰화 및 DataLoader 구성 완료!")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print("첫 문장 예시:", next(iter(train_loader))["input_ids"][0][:50])
    print("classifier 단독 실행 중")

