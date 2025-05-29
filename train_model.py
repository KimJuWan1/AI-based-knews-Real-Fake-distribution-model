# =============================
# [1] 라이브러리 및 모듈 임포트
# =============================
import torch
import torch.nn as nn
from transformers import AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from classifier import get_dataloaders
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# =============================
# [2] 평가 지표 및 시각화 함수 정의-각 에포크마다 성능을 확인할 수 있게 설정
# =============================
def evaluate_competition(y_true, y_pred, y_probs=None, class_names=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}")                    #Accuracy계산
    print(f"Precision (macro): {precision:.4f}")     #Precision계산
    print(f"Recall (macro): {recall:.4f}")           #Recall계산
    print(f"F1-score (macro): {f1:.4f}")             #F1-score계산

# =============================
# [3] 모델 정의
# =============================
class DebertaClassifier(nn.Module):   #PTorch의 nn.module상속, 사용자 정의 모델 클래스 만듬:
    def __init__(self, pretrained_model="microsoft/deberta-v3-base", hidden_size=768, dropout_rate=0.3, num_classes=4):   #기본값 deberta-v3-base사용,출력차원:4(클래스 분류를 위한 설정)
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)      #AutoModel을 사용해 DeBERTa-v3기반 Transformer인코더 불러오기, 출력은 hidden state전체(batch_size X seq_len X hidden_dim), 여기서 [CLS]토큰에 해당하는 첫 토큰 출력만 사용
        self.dropout = nn.Dropout(dropout_rate)                         #과적합 방지를 위한 dropout적용, classification head에 들어가기 전 [CLS]벡터에 적용
        self.classifier = nn.Linear(hidden_size, num_classes)           #hidden state크기(768)-> 클래스 4로 매핑

    def forward(self, input_ids, attention_mask, labels=None):           #순전파 진행forward()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)  #Transformer 인코더 실행, outputs.last_hidden_state는 전체 시퀀스의 출력(batch, seq_len, hidden)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])               #시퀀스 첫 번째 토큰 [:,0,:]을 추출-> [CLS]역할 그 후 dropout 적용
        logits = self.classifier(cls_output)                                        #각 클래스에 대한 logit 점수 (softmax 전단계)
        
        if labels is not None:                                                     #학습 시: 손실 계산
            loss_fct = nn.CrossEntropyLoss()                                       #학습 단계: 정답 labels가 주어졌다면 CrossEntropyLoss로 손실계산
            loss = loss_fct(logits, labels)                                        #(loss, logits)반환하여 학습 루프에서 사용
            return loss, logits
        else:
            return logits                                                           #평가/추론 시에는 손실이 필요 없으므로 logits만 반환


# =============================
# [4] 학습 루프
# =============================

def train_model(train_loader, val_loader, num_epochs=32, lr=2e-5, weight_decay=0.01, patience=100):
    #학습환경 설정 및 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #학습환경의 gpu가용여부 판단하여 연산 디바이스 설정
    print(f"사용 디바이스: {device}")
    model = DebertaClassifier().to(device)

    if os.path.exists("best_model.pt"):                           #기존에 저장된 최적 가중치 파일(best_model.pt)이 존재하는 경우, 해당 모델을 로딩하여 이어서 학습가능
        print("기존 best_model.pt 불러오는 중...")
        model.load_state_dict(torch.load("best_model.pt"))
    else:                                                          #그 외의 경우 새로운 모델 인스턴스를 생성하여 학습 시작
        print("새로운 모델로 학습 시작")
    
    #Optimizer및 학습 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)     #최적화 함수 AdamW사용,학습률:2e-5, weight decay:0.01
    total_steps = len(train_loader) * num_epochs                                              
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)  #Warmup을 포함한 linear scheduler를 적용하여 전체 스테수에 비례하여 learning rate를점진적으로 조정

    best_f1 = 0
    no_improve = 0
    class_names = ["HF", "HR", "MF", "MR"]
    
    #epoch기반 반복학습 진행-각 Epoch마다 train_loader를 통해 배치 단위로 모델을 학습. 학습 시 각 배치마다 gradient 초기화 → forward 연산 → loss 계산 → backward → gradient clipping → optimizer 업데이트 → scheduler 업데이트 순서로 진행.
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        # epoch종료후 validation 평가수행-각 Epoch이 종료된 후, validation 데이터셋(val_loader)을 통해 현재 모델의 성능을 평가. 예측된 결과로부터 Macro F1-score, Accuracy, Precision, Recall 등의 성능 지표를 출력.

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.append(probs.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"\n Epoch {epoch+1} | Macro F1: {f1:.4f}")
        evaluate_competition(all_labels, all_preds, np.vstack(all_probs), class_names)

        #best model 저장 로직-현재 Epoch에서 계산된 macro F1-score가 이전까지의 최고 성능보다 우수할 경우, 해당 모델 가중치를 best_model.pt로 저장. 그렇지 않다면 no_improve 카운터를 증가시켜 Early Stopping을 위한 준비. 기준값:100
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            #torch.save(model.state_dict(), "best_model.pt")
            torch.save(model.state_dict(), "best_model_more train.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️ Early stopping!")
                break
    #최종반환-최종적으로 학습이 완료된 모델 인스턴스를 반환하며, 이후 별도 평가 모듈(analysis.py)에서 "best_model.pt"경우의 validation 및 test 데이터셋에 대한 성능 평가를 수행.
    return model

"""
# =============================
# [5] 추론
# =============================
def inference(model, test_loader, save_path="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())

    df = pd.DataFrame({"id": list(range(len(preds))), "label": preds})
    df.to_csv(save_path, index=False)
    print(f"📦 submission saved to {save_path}")
"""

# =============================
# [6] 실행 엔트리포인트
# =============================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    model = train_model(train_loader, val_loader)
    model.load_state_dict(torch.load("best_model.pt"))
    #inference(model, test_loader)
