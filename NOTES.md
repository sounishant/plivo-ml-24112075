1. Our model relies heavily on linguistic turn-yielding bundles, combining creaky voice periodicity, register-normalized F0 percentiles, and energy distance relative to the speaker's silence floor.
2. Causal constraints are strictly enforced by computing all historical baselines using only audio preceding `pause_start`.
3. Initial linear models suffered from probability compression on Hindi data, pinning performance to an 850ms fallback delay.
4. Transitioning from standard AUC splitting to a custom GroupKFold cross-validation loop optimizing directly against score.py solved the objective mismatch.
5. Tree-based non-linear mapping via a tuned Random Forest captured feature interactions without overfitting to the 496 training pauses.
6. Short-hold sample weighting successfully protected the model against destructive false cutoffs that ruin the 5% error budget.
7. The model occasionally still fails on abrupt speaker disfluencies where mid-sentence pauses mimic phrase-final lengthening.
8. With one more day, we would incorporate delta-features (velocity and acceleration of pitch and energy) to capture finer temporal inflection curves.
9. We would also implement per-language bias calibration parameters to align threshold distributions natively across English and Hindi.
10. Finally, exploring lightweight sequence models over multi-pause turn contexts would provide stronger long-term conversational memory.