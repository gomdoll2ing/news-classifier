import gc
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 5단계에서 쓰던 함수 그대로
def map_to_main_sector(sector_str):
    if not isinstance(sector_str, str):
        return '기타'
    if sector_str in ['경제', '부동산', '돈 버는 재미', '머니랩']:
        return '경제'
    if sector_str in ['정치', '더 북한']:
        return '정치'
    if sector_str in ['사회', '피플', '세상과 함께', '가족과 함께', 'hello! Parents', '톡톡에듀']:
        return '사회'
    if sector_str in ['국제', '더 차이나']:
        return '국제'
    if sector_str in ['스포츠', '문화', '여행레저', 'COOKING', '쉴 땐 뭐하지',
                      '더,마음', '2024 파리올림픽', '더,오래', '더 헬스', '라이프',
                      '마음 챙기기', '더 하이엔드', '비크닉']:
        return '문화/스포츠'
    if sector_str in ['브랜드뉴스', '오피니언', '중앙SUNDAY', 'Leader & Reader', 'Tran:D']:
        return '기타'
    return '기타'

def time_split(raw_dataset, col_date='yyyymmdd', col_label='sector1'):
    meta_df = raw_dataset.select_columns([col_date, col_label]).to_pandas()
    valid_idx = meta_df[
        (meta_df[col_date].notna()) &
        (meta_df[col_label].notna()) &
        (meta_df[col_date].astype(float) >= 20200101) &
        (meta_df[col_date].astype(float) <= 20251231)
    ].index.to_numpy()

    sorted_idx = valid_idx[np.argsort(meta_df.loc[valid_idx, col_date].values)]
    total_len = len(sorted_idx)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)

    train_idx = sorted_idx[:train_end]
    val_idx   = sorted_idx[train_end:val_end]
    test_idx  = sorted_idx[val_end:]

    del meta_df, valid_idx, sorted_idx
    gc.collect()
    return train_idx, val_idx, test_idx

def build_label_encoder(raw_dataset, train_idx):
    train_labels_raw = raw_dataset.select(train_idx)['sector1']
    agg_labels = [map_to_main_sector(s) for s in train_labels_raw]
    le = LabelEncoder()
    le.fit(agg_labels)
    return le
