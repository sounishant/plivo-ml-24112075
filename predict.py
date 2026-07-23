"""python predict.py --data_dir eot_data/hindi --out predictions.csv"""
import argparse
import csv
import joblib
from feat_eng import build_features_only

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args()

    clf = joblib.load(args.model)
    X, keys = build_features_only(args.data_dir)
    p = clf.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")

if __name__ == "__main__":
    main()