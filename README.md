# AI-based-knews-Real-Fake-distribution-model

📰 AI 기반 뉴스 진위/출석 분류 모델 (DeBERTa-v3)

📌 프로젝트 개요

이 프로젝트는 뉴스 기사 텍스트를 기반으로
4가지 클래스로 분류하는 DeBERTa-v3 기반 분류 모델입니다.

분류 클래스는 다음과 같습니다:

HF: Human-generated Fake

HR: Human-generated Real

MF: Machine-generated Fake

MR: Machine-generated Real

🛠️ 사용 기술

모델: microsoft/deberta-v3-base

프레임워크: PyTorch, HuggingFace Transformers

환경: Python 3.10+, CUDA GPU 지원 (권장)

📁 폴더 구조

project/
├── classifier.py           # 데이터셋 로딩 및 전체 전처리
├── train_model.py          # 모델 정의 및 학습 루프
├── evaluate.py             # 평가 및 confusion matrix 시각화
├── requirements.txt        # 의종 패키지 목록
├── best_model.pt           # 학습된 차택 성능 모델
└── README.md               # 프로젝트 설명 문서

🚀 실행 방법

의종 패키지 설치

pip install -r requirements.txt

학습 실행

python train_model.py

평가 및 시각화

python evaluate.py

출력적인 submission.csv 생성

# train_model.py 내 inference 함수 및 test_loader 발행


🤖 추리 구조

입력 뉴스 텍스트 → AutoTokenizer 통해 토크나이즈 (max_length=512, padding/truncation)

DeBERTa encoder → [CLS] 베터 추출 → Dropout → Linear Classifier → Softmax

최종 예측: argmax(softmax(logits)) → 클래스 인덱스 첫지

📌 데이터 구성

Train:Validation:Test = 8:1:1 (학습 속도와 효율 고보)

Label은 보고치 4가지 분류 타입 (0~3)



