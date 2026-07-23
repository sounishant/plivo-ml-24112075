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
"""python train.py --data_dir eot_data/english eot_data/hindi --out model.joblib"""
"""python train.py --data_dir eot_data/english eot_data/hindi --out model.joblib"""
import argparse
import joblib
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feat_eng import build_dataset
from score import evaluate, THRESHOLDS, DELAYS

warnings.filterwarnings('ignore')

def oof_predict(X, y, groups, durations, C, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        # Heavily penalize errors on short holds (the ones that ruin the score budget)
        hold_short = (y[tr] == 0) & (durations[tr] < np.median(durations[tr][y[tr] == 0]))
        w = np.where(hold_short, 3.0, 1.0)   
        
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                               max_iter=2000, class_weight="balanced", random_state=42))
        clf.fit(X[tr], y[tr], logisticregression__sample_weight=w)
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof

def real_metric(oof_p, y, durations, groups, budget=0.05):
    pauses = [{"turn_id": g, "dur": d, "label": "eot" if l else "hold", "p": p}
              for g, d, l, p in zip(groups, durations, y, oof_p)]
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d)
            if cut <= budget and (best is None or lat < best[0]):
                best = (lat, cut, t, d)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", required=True)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    X, y, groups, keys, durations = build_dataset(args.data_dir)
    print(f"pooled: {len(y)} pauses, {len(set(groups))} turns")

    # Pick the best hyperparameters based on the REAL delay metric, not AUC
    best_c, best_lat = None, float('inf')
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        oof = oof_predict(X, y, groups, durations, C)
        metric_res = real_metric(oof, y, durations, groups)
        if metric_res:
            lat, cut, t, d = metric_res
            print(f"C={C:<6} OOF mean_delay={lat*1000:.0f}ms  cutoff={cut*100:.1f}%")
            if lat < best_lat:
                best_lat = lat
                best_c = C
        else:
            print(f"C={C:<6} Failed to find valid operating point <= 5% cutoff")

    print(f"\nTraining final model with best C={best_c}...")
    
    # Train final shipped model on all data with the best C
    hold_short = (y == 0) & (durations < np.median(durations[y == 0]))
    w = np.where(hold_short, 3.0, 1.0)
    
    final_clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=best_c, penalty="l1", solver="liblinear",
                                           max_iter=2000, class_weight="balanced", random_state=42))
    final_clf.fit(X, y, logisticregression__sample_weight=w)
    joblib.dump(final_clf, args.out)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()