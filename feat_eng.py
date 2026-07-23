"""Shared feature engineering. train.py and predict.py both import this,
so training and inference always compute features identically."""
import csv
import os
from collections import defaultdict
import numpy as np
from features import load_wav, speech_before, frame_energy_db, f0_contour

N_FEATURES = 13

def extract_features(x, sr, pause_start, prior_durations):
    """Causal features from audio strictly before `pause_start`.
    prior_durations = durations of EARLIER pauses in this turn (already
    finished before pause_start -> legal to use)."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # speaker's own running baseline, from all causal audio so far
    hist = x[:int(pause_start * sr)]
    hist_f0 = f0_contour(hist, sr) if len(hist) > sr // 5 else np.array([0.0])
    hist_voiced = hist_f0[hist_f0 > 0]
    base_f0 = float(np.median(hist_voiced)) if len(hist_voiced) else 0.0
    hist_e = frame_energy_db(hist, sr) if len(hist) > sr // 5 else np.array([-60.0])
    base_e = float(np.median(hist_e))

    # pitch in the pre-pause window
    f0 = f0_contour(seg, sr)
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    if len(voiced) >= 4:
        final_f0 = float(np.mean(voiced[-3:]))
        t = np.arange(len(f0))[voiced_mask]
        f0_slope = float(np.polyfit(t, voiced, 1)[0])
    else:
        final_f0, f0_slope = 0.0, 0.0
    f0_rel = (final_f0 - base_f0) if (base_f0 > 0 and final_f0 > 0) else 0.0

    # energy trajectory
    e = frame_energy_db(seg, sr)
    final_e = float(e[-5:].mean())
    e_slope = float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) >= 4 else 0.0
    e_rel = final_e - base_e

    # final-syllable lengthening proxy: last voiced run vs. average voiced run
    seg_lens, run = [], 0
    for v in voiced_mask:
        if v:
            run += 1
        elif run:
            seg_lens.append(run); run = 0
    if run:
        seg_lens.append(run)
    last_voiced_len = seg_lens[-1] if seg_lens else 0
    mean_voiced_len = float(np.mean(seg_lens)) if seg_lens else 0.0
    lengthening_ratio = (last_voiced_len / mean_voiced_len) if mean_voiced_len > 0 else 0.0

    # speaking-rate / voicing proxies
    voicing_frac = float(voiced_mask.mean())
    rate = len(seg_lens) / (len(seg) / sr)

    # turn-so-far context (all causal: these pauses already ended)
    n_prior = len(prior_durations)
    mean_prior_dur = float(np.mean(prior_durations)) if prior_durations else 0.0
    max_prior_dur = float(np.max(prior_durations)) if prior_durations else 0.0

    return np.array([
        final_f0, f0_slope, f0_rel, final_e, e_slope, e_rel,
        lengthening_ratio, voicing_frac, rate,
        n_prior, mean_prior_dur, max_prior_dur, len(seg) / sr,
    ], dtype=np.float32)

def _iter_rows_with_context(data_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)
    cache = {}
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: int(r["pause_index"]))
        prior_durations = []
        for r in turn_rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            feats = extract_features(x, sr, float(r["pause_start"]), prior_durations)
            yield r, feats
            prior_durations.append(float(r["pause_end"]) - float(r["pause_start"]))

def build_dataset(data_dirs):
    X, y, groups, keys = [], [], [], []
    for d in data_dirs:
        for r, feats in _iter_rows_with_context(d):
            X.append(feats)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(r["turn_id"])
            keys.append((r["turn_id"], r["pause_index"]))
    return np.array(X), np.array(y), groups, keys

def build_features_only(data_dir):
    X, keys = [], []
    for r, feats in _iter_rows_with_context(data_dir):
        X.append(feats)
        keys.append((r["turn_id"], r["pause_index"]))
    return np.array(X), keys