import os
import sys

# 프로젝트 루트 잡기
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation.gpt_labeled_eval import evaluate_gpt_labeled_file

if __name__ == "__main__":
    # 기본은 1000개짜리 파일
    evaluate_gpt_labeled_file("chatgpt_1000_labels_1990_2019.parquet")
