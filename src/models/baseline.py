import pickle
import gc
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from config.config import SAVE_BASE_PATH

LR_MODEL_PATH = SAVE_BASE_PATH / "baseline_lr.pkl"
LR_VEC_PATH = SAVE_BASE_PATH / "baseline_tfidf_vec.pkl"
LR_REPORT_PATH = SAVE_BASE_PATH / "baseline_report_lr.pkl"

def train_or_load_baseline(train_ds_processed, test_ds_processed, label_encoder):
    X_test_text = test_ds_processed['text']
    y_test = test_ds_processed['label']

    if LR_MODEL_PATH.exists() and LR_VEC_PATH.exists() and LR_REPORT_PATH.exists():
        with open(LR_VEC_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(LR_MODEL_PATH, 'rb') as f:
            lr_clf = pickle.load(f)
        with open(LR_REPORT_PATH, 'rb') as f:
            report_lr = pickle.load(f)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
        y_pred = lr_clf.predict(X_test_tfidf)
    else:
        X_train_text = train_ds_processed['text']
        y_train = train_ds_processed['label']

        tfidf_vectorizer = TfidfVectorizer(min_df=5, max_features=50000, sublinear_tf=True)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

        lr_clf = LogisticRegression(
            solver='saga',
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        lr_clf.fit(X_train_tfidf, y_train)

        with open(LR_VEC_PATH, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(LR_MODEL_PATH, 'wb') as f:
            pickle.dump(lr_clf, f)

        del X_train_tfidf, X_train_text, y_train
        gc.collect()

        y_pred = lr_clf.predict(X_test_tfidf)

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    with open(LR_REPORT_PATH, 'wb') as f:
        pickle.dump(report, f)

    return report
