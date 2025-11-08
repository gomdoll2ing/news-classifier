import shutil
import time
from config.config import DATA_PATH, ALT_DATA_PATH, RAM_DATA_PATH

def mount_drive():
    """
    Colab 노트북 환경일 때만 드라이브 마운트하고,
    그게 아니면 조용히 스킵한다.
    """
    try:
        # 여기서만 google.colab을 import 한다
        from google.colab import drive  # 이 줄이 위로 나가면 안 됨
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted")
    except ModuleNotFoundError:
        # !python scripts/train.py 처럼 노트북 밖에서 실행될 때
        print("ℹ️ Colab 환경이 아니라서 드라이브 마운트를 건너뜁니다.")
    except Exception as e:
        # 혹시 다른 이유로도 실패하면 그냥 알리고 넘어가자
        print(f"⚠️ 드라이브 마운트 실패, 계속 진행합니다: {e}")

def copy_to_ram():
    # 어느 경로든 실제로 있는 parquet를 고른다
    src = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    if not src.exists():
        raise FileNotFoundError("combined_articles.parquet 를 찾을 수 없습니다.")

    if not RAM_DATA_PATH.exists():
        start = time.time()
        shutil.copyfile(src, RAM_DATA_PATH)
        print(f"✅ Drive → RAM 디스크 복사 완료 ({time.time() - start:.1f}s)")
    else:
        print("✅ RAM 디스크에 이미 있음 → 복사 생략")

def load_raw_dataset():
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": str(RAM_DATA_PATH)}, split="train")
    print(f"✅ Dataset loaded: {len(ds):,} rows")
    return ds
