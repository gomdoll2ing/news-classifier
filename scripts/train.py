import sys
import os

# ===== 경로 세팅 부분 (중요!) =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ================================

from config.config import N_PROC
from src.data.loader import mount_drive, copy_to_ram, load_raw_dataset
from src.preprocess.label_map import (
    time_split,
    build_label_encoder,
    map_to_main_sector,
)
from src.models.baseline import train_or_load_baseline

def preprocess_dataset(raw, idx, label_encoder):
    subset = raw.select(idx)
    def _pp(examples):
        h = [t if t else "" for t in examples['headline']]
        c = [t if t else "" for t in examples['content']]
        texts = [f"{hh} {cc}"[:1000] for hh, cc in zip(h, c)]
        main = [map_to_main_sector(s) for s in examples['sector1']]
        labels = label_encoder.transform(main)
        return {"text": texts, "label": labels}
    return subset.map(_pp, batched=True, num_proc=N_PROC, remove_columns=raw.column_names)

def main():
    print("📁 1. 드라이브 마운트")
    mount_drive()
    print("📦 2. RAM 디스크로 복사")
    copy_to_ram()
    print("📚 3. 데이터셋 로드")
    raw = load_raw_dataset()

    print("📅 4. 시계열 분할")
    train_idx, val_idx, test_idx = time_split(raw)
    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    print("🏷️ 5. 라벨 인코더 생성")
    label_encoder = build_label_encoder(raw, train_idx)

    print("🧪 6. 전처리 적용")
    train_ds_processed = preprocess_dataset(raw, train_idx, label_encoder)
    val_ds_processed   = preprocess_dataset(raw, val_idx, label_encoder)
    test_ds_processed  = preprocess_dataset(raw, test_idx, label_encoder)

    print("🧪 7. 베이스라인 학습/로드")
    report_lr = train_or_load_baseline(train_ds_processed, test_ds_processed, label_encoder)
    print("=== Baseline (TF-IDF + LR) Weighted avg ===")
    print(report_lr["weighted avg"])

if __name__ == "__main__":
    main()
