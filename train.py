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

import argparse
import joblib
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from feat_eng import build_dataset
from score import evaluate, THRESHOLDS, DELAYS

warnings.filterwarnings('ignore')

def oof_predict(X, y, groups, durations, n_estimators, max_depth, min_samples_leaf, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        hold_short = (y[tr] == 0) & (durations[tr] < np.median(durations[tr][y[tr] == 0]))
        w = np.where(hold_short, 3.0, 1.0)   
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X[tr], y[tr], sample_weight=w)
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

    best_params, best_lat = None, float('inf')
    
    # Expanded Grid Search
    for n_est in [100, 150, 200, 300]:
        for depth in [3, 4, 5, 6]:
            for min_leaf in [5, 10, 15]:
                oof = oof_predict(X, y, groups, durations, n_est, depth, min_leaf)
                metric_res = real_metric(oof, y, durations, groups)
                if metric_res:
                    lat, cut, t, d = metric_res
                    print(f"n_est={n_est} depth={depth} leaf={min_leaf} | OOF mean_delay={lat*1000:.0f}ms cutoff={cut*100:.1f}%")
                    if lat < best_lat:
                        best_lat = lat
                        best_params = (n_est, depth, min_leaf)

    print(f"\nTraining final Random Forest model with best params: {best_params}...")
    
    hold_short = (y == 0) & (durations < np.median(durations[y == 0]))
    w = np.where(hold_short, 3.0, 1.0)
    
    final_clf = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_leaf=best_params[2],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_clf.fit(X, y, sample_weight=w)
    joblib.dump(final_clf, args.out)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()