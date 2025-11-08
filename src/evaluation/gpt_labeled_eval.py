from pathlib import Path
import pickle
import gc
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from config.config import SAVE_BASE_PATH

def load_artifacts(save_base_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DL 모델
    dl_model = AutoModelForSequenceClassification.from_pretrained(save_base_path).to(device)

    # 2) 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(save_base_path)

    # 3) 라벨 인코더
    with open(save_base_path / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # 4) LR 모델
    with open(save_base_path / "baseline_lr.pkl", "rb") as f:
        lr_clf = pickle.load(f)

    # 5) TF-IDF 벡터라이저
    with open(save_base_path / "baseline_tfidf_vec.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    return {
        "device": device,
        "dl_model": dl_model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "lr_clf": lr_clf,
        "tfidf_vectorizer": tfidf_vectorizer,
    }

def load_gpt_labeled_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"라벨링된 parquet 를 찾을 수 없습니다: {path}")
    df = pd.read_parquet(path)
    return df

def run_lr_inference(df: pd.DataFrame, tfidf_vectorizer, lr_clf, label_encoder):
    # 원래 전처리랑 똑같이: headline + content 잘라서 1000자
    texts = df.apply(
        lambda row: f"{row.get('headline','') or ''} {row.get('content','') or ''}"[:1000],
        axis=1
    )
    X = tfidf_vectorizer.transform(texts)
    pred_idx = lr_clf.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    df["lr_label"] = pred_labels
    return df

def _tokenize_batch(examples, tokenizer):
    h = [t if t else "" for t in examples["headline"]]
    c = [t if t else "" for t in examples["content"]]
    texts = [f"{hh} {cc}"[:1000] for hh, cc in zip(h, c)]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

def run_dl_inference(df: pd.DataFrame, dl_model, tokenizer, label_encoder, device, batch_size: int = 96):
    ds = Dataset.from_pandas(df)

    ds_tok = ds.map(
        lambda e: _tokenize_batch(e, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=ds.column_names,
    )
    ds_tok.set_format("torch")

    args = TrainingArguments(
        output_dir="/content/temp_inference_output",
        per_device_eval_batch_size=batch_size,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
    )
    trainer = Trainer(model=dl_model, args=args)
    pred_output = trainer.predict(ds_tok)
    pred_idx = np.argmax(pred_output.predictions, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    df["dl_label"] = pred_labels
    return df

def compare_with_gpt(df: pd.DataFrame, save_base_path: Path):
    # GPT가 제대로 뱉은 것만 비교 (분류실패/분류오류 제외)
    valid = df[
        (df["chatgpt_label"] != "분류실패")
        & (df["chatgpt_label"] != "분류오류")
    ].copy()

    if len(valid) == 0:
        print("⚠️ 유효한 GPT 라벨이 없습니다.")
        return

    lr_acc = accuracy_score(valid["chatgpt_label"], valid["lr_label"])
    dl_acc = accuracy_score(valid["chatgpt_label"], valid["dl_label"])

    print(f"총 샘플: {len(df)}개")
    print(f"유효 샘플: {len(valid)}개")
    print(f"[LR]  GPT 라벨 기준 정확도: {lr_acc:.2%}")
    print(f"[DL]  GPT 라벨 기준 정확도: {dl_acc:.2%}")

    # 결과 저장
    out_path = save_base_path / f"final_{len(df)}_comparison_result.parquet"
    valid.to_parquet(out_path, index=False)
    print(f"✅ 결과 저장: {out_path}")

def evaluate_gpt_labeled_file(parquet_name: str = "chatgpt_1000_labels_1990_2019.parquet"):
    save_base_path = SAVE_BASE_PATH
    artifacts = load_artifacts(save_base_path)

    df = load_gpt_labeled_parquet(save_base_path / parquet_name)

    # 1) LR
    df = run_lr_inference(
        df,
        artifacts["tfidf_vectorizer"],
        artifacts["lr_clf"],
        artifacts["label_encoder"],
    )

    # 2) DL
    df = run_dl_inference(
        df,
        artifacts["dl_model"],
        artifacts["tokenizer"],
        artifacts["label_encoder"],
        artifacts["device"],
    )

    # 3) 비교 + 저장
    compare_with_gpt(df, save_base_path)

    # 메모리 클린업
    del df
    gc.collect()
