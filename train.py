"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv

Ideas worth testing (this is the assignment, not a checklist):
  - F0 slope over the last voiced region (statements fall, continuations
    often stay level or rise)
  - final-syllable lengthening: last voiced stretch duration vs the
    speaker's average
  - energy decay rate into the pause
  - speaking-rate context, position of the pause within the turn so far
  - anything you discover by LISTENING to your misclassified pauses
"""
"""python train.py --data_dir eot_data/english eot_data/hindi --out model.joblib"""
"""python train.py --data_dir eot_data/english eot_data/hindi --out model.joblib"""
import argparse
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from feat_eng import build_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", required=True)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    X, y, groups, keys = build_dataset(args.data_dir)
    print(f"pooled: {len(y)} pauses, {len(set(groups))} turns, {len(args.data_dir)} folder(s)")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
    
    # UPGRADED CLASSIFIER HERE
    clf = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42))
    
    clf.fit(X[tr], y[tr])
    p_te = clf.predict_proba(X[te])[:, 1]
    print(f"held-out turn AUC: {roc_auc_score(y[te], p_te):.3f}  (0.5=chance, 1.0=perfect)")

    clf.fit(X, y)  # Refit on all available data for the shipped model
    joblib.dump(clf, args.out)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()