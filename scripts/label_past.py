import sys, os
from pathlib import Path

# 프로젝트 루트 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.config import SAVE_BASE_PATH, DRIVE_ROOT
from src.labeling.gpt_labeler import (
    load_past_dataset,
    pick_new_samples,
    label_batch,
    save_merged,
)

def main():
    # 1. 경로 정의
    original_data_path = DRIVE_ROOT / "joongang_crawl" / "combined_articles.parquet"
    if not original_data_path.exists():
        alt = DRIVE_ROOT / "combined_articles.parquet"
        if not alt.exists():
            raise FileNotFoundError("원본 parquet를 찾을 수 없습니다.")
        original_data_path = alt

    local_ram_path = Path("/dev/shm/combined_articles.parquet")

    existing_500_path = SAVE_BASE_PATH / "chatgpt_500_labels_1990_2019.parquet"
    final_1000_path = SAVE_BASE_PATH / "chatgpt_1000_labels_1990_2019.parquet"

    # 2. 과거 데이터 로드
    past_ds = load_past_dataset(original_data_path, local_ram_path)
    print(f"1990~2019 데이터: {len(past_ds):,}건")

    # 3. 신규 샘플 뽑기
    df_new_samples = pick_new_samples(past_ds, None, start=500, end=1000)
    print(f"신규 샘플: {len(df_new_samples)}건")

    # 4. GPT 라벨링
    df_labeled = label_batch(df_new_samples, max_workers=10, model="gpt-4o-mini")

    # 5. 병합 저장
    merged = save_merged(existing_500_path, df_new_samples, df_labeled, final_1000_path)
    print(f"최종 저장 완료: {final_1000_path} ({len(merged)}건)")

if __name__ == "__main__":
    main()
