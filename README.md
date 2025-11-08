# 📰 News Classifier (Joongang Crawl 기반)

한국어 뉴스 기사(중앙일보 크롤링 데이터)를 6개 메인 카테고리로 분류하는 파이프라인입니다.

- **학습 데이터**: 2020~2025년 뉴스 (라벨 있음)
- **테스트/확장 데이터**: 1990~2019년 뉴스 (ChatGPT API로 라벨 생성)
- **주요 라벨**: `경제`, `정치`, `사회`, `국제`, `문화/스포츠`, `기타`

---

## 1. 프로젝트 구조

```text
news-classifier/
├── README.md
├── config/
│   └── config.py             # 경로 등 공통 설정
├── src/
│   ├── data/
│   │   └── loader.py         # Drive → RAM 복사 및 parquet 로드
│   ├── preprocess/
│   │   └── label_map.py      # 섹터 → 6개 대분류 매핑, 시계열 split
│   ├── models/
│   │   └── baseline.py       # TF-IDF + Logistic Regression 베이스라인
│   ├── labeling/
│   │   └── gpt_labeler.py    # OpenAI API 기반 과거 기사 라벨링
│   └── utils/                # (필요시) 공통 유틸
├── scripts/
│   ├── train.py              # 학습 전체 파이프라인
│   └── label_past.py         # 1990~2019 뉴스 자동 라벨링
└── notebooks/                # 실험/EDA 용도
```

## 2. 실행 방법
### 2-1. 학습 파이프라인

Colab 기준:
```bash
%cd /content/news-classifier
!python scripts/train.py
```

이 스크립트는 다음을 수행합니다.
1. (가능하면) 드라이브에서 데이터 읽기
2. /dev/shm(RAM 디스크)로 parquet 복사
3. yyyymmdd 기준 시계열 분할 (train/val/test)
4. 세부 섹터를 6개 메인 라벨로 집계
5. TF-IDF + 로지스틱 회귀 학습 및 저장

### 2-2. 과거 뉴스 라벨링 파이프라인
```bash
%cd /content/news-classifier
!python scripts/label_past.py
```

이 스크립트는 다음을 수행합니다.
1. 전체 parquet에서 1990~2019년 기사만 필터
2. 하루치(예: 500건) 샘플링
3. OpenAI API로 라벨 생성
4. 기존에 저장해 둔 라벨링 결과와 병합하여 Drive에 parquet로 저장

## 3. 환경 변수 (OpenAI API 키)

Colab에서:
```bash
import os
os.environ["OPENAI_API_KEY"] = "sk-...당신의키..."
Colab의 “비밀 변수”에 저장해두고 userdata.get(...)로 꺼내 써도 됩니다.
```

## 4. Colab에서 개발 루틴
# 1) 세션 시작
```bash
%cd /content
!git clone https://github.com/<YOUR_ID>/news-classifier.git
%cd /content/news-classifier
```

# 2) 코드 수정
```bash
%%writefile src/models/baseline.py
# ...수정한 코드...
```

# 3) 테스트
```bash
!python scripts/train.py
```

# 4) GitHub 반영
```bash
!git add .
!git commit -m "feat: add labeling script"
!git push origin main
```

## 5. 향후 계획 (Roadmap)
1. KLUE-RoBERTa 파인튜닝 코드 추가 (src/models/roberta.py)
2. GPT 라벨 품질 점검 노트북 추가
3. 라벨링 배치 크기 파라미터화
4. Docker / 로컬 실행 스크립트화

## 6. License
개인 연구 및 포트폴리오 용도.

1. 콜랩에서 이 내용으로 파일 만들기

```bash
%%writefile /content/news-classifier/README.md
# 📰 News Classifier ...
(위에 있는 내용 전부 붙여넣기)
```

2. git에 반영
```bash
%cd /content/news-classifier
!git add README.md
!git commit -m "docs: add README"
!git push origin main
```
