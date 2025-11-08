from pathlib import Path

# 공통 경로
DRIVE_ROOT = Path("/content/drive/MyDrive")

# 원본 데이터 위치 (둘 중 하나 있는 걸로 씀)
DATA_PATH = DRIVE_ROOT / "joongang_crawl" / "combined_articles.parquet"
ALT_DATA_PATH = DRIVE_ROOT / "combined_articles.parquet"

# 모델/리포트 저장 위치
SAVE_BASE_PATH = DRIVE_ROOT / "best_news_classifier"
SAVE_BASE_PATH.mkdir(parents=True, exist_ok=True)

# RAM 디스크 위치
RAM_DATA_PATH = Path("/dev/shm/combined_articles.parquet")

# 멀티프로세싱 개수
N_PROC = 16
