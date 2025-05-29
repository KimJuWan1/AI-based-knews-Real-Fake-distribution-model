import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import get_dataloaders
from train_model import DebertaClassifier
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로딩
train_loader, val_loader, test_loader = get_dataloaders()

# 모델 로딩
model = DebertaClassifier().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

def evaluate_model(loader, dataset_name="Validation"):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print(f"\n [{dataset_name}] 평가 결과")
    report = classification_report(
        all_labels, all_preds,
        target_names=["HF", "HR", "MF", "MR"],
        digits=4,
        output_dict=True
    )
    print(classification_report(all_labels, all_preds, target_names=["HF", "HR", "MF", "MR"], digits=4))

    # MF 클래스 성능만 추출
    mf_precision = report["MF"]["precision"]
    mf_recall = report["MF"]["recall"]
    mf_f1 = report["MF"]["f1-score"]

    print(f" [MF] Precision: {mf_precision:.4f}")
    print(f" [MF] Recall:    {mf_recall:.4f}")
    print(f" [MF] F1-score:  {mf_f1:.4f}")

    # Confusion matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HF", "HR", "MF", "MR"], yticklabels=["HF", "HR", "MF", "MR"])
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # MF 성능 바 차트
    plt.figure(figsize=(4, 4))
    plt.bar(["Precision", "Recall", "F1-score"], [mf_precision, mf_recall, mf_f1])
    plt.ylim(0, 1)
    plt.title(f"{dataset_name} - MF detection performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Validation 평가
    evaluate_model(val_loader, dataset_name="Validation")

    # Test 평가
    evaluate_model(test_loader, dataset_name="Test")

     # 클래스 분포 시각화 추가
    from classifier import load_data

    df = load_data()
    label_map = {0: "HR", 1: "HF", 2: "MR", 3: "MF"}
    df["label_name"] = df["label"].map(label_map)

    print("\n 클래스별 샘플 수:")
    print(df["label_name"].value_counts())

    df["label_name"].value_counts().sort_index().plot(
        kind="bar", color="lightblue", edgecolor="black", title="number of samples per class"
    )
    plt.xlabel("Class")
    plt.ylabel("number of samples")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()