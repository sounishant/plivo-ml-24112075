# RUNLOG.md

- **Run 1 (Baseline English)**: Score = 1600 ms delay @ <=5% cutoff (AUC: 0.514)
- **Run 1 (Baseline Hindi)**: Score = 850 ms delay @ <=5% cutoff (AUC: 0.501)

- **Run 2 (Prosodic Features + Logistic Regression)**:
  - English: 1345 ms delay (AUC: 0.697) -> 255ms improvement!
  - Hindi: 850 ms delay (AUC: 0.682) -> AUC improved, but delay is still the same.

  - **Run 3 (HistGradientBoosting, overfitted)**:
  - English: 100 ms delay (AUC 1.000)
  - Hindi: 100 ms delay (AUC 1.000)
  - Note: The held out AUC during training dropped to 0.631. The 1.000 AUC is artificial and it happened due to the tree perfectly memorizing the complete data as the training set is small enough to be memorized completely by a tree (496 samples) during the final full refit. 

  - **Run 4 (Linguistic Features + CV Loss Optimization)**:
  - English: 1384 ms (AUC: 0.702)
  - Hindi: 850 ms (AUC: 0.721)
  - *Note*: Added periodicity and relative pitch/energy features, replaced AUC validation loop with a GroupKFold validation loop optimizing directly against the score.py evaluation metric with short hold sample weighting and also used L1 regularization.