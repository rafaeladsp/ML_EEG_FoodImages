# MVPA regression of food ratings

Predict continuous healthiness, tastiness and willingness-to-eat ratings from EEG.
This complements the binary classification analyses in the final notebooks.

## Run

Open `mvpa_svr_boosting.ipynb` from the repository root or the `mvpa` folder.
The first cell installs the dependencies. Set the data and ratings paths in
`config` before running. By default the EEG file is `~/Downloads/epoch_data.nc`
and ratings are local `analysis35/ratings/FOODEEG_pXXX_ratings.csv` files.
These data are not included in Git. Participant IDs are in
`analysis35/subjects.csv` (35 participants selected for the earlier analysis).

The loader reads one participant at a time using xarray. It currently uses
**display epochs only**, from -100 to 1000 ms. It averages the three presentations
of each image, giving 120 unique-image examples per participant.

## Protocol

- Benchmark: mean EEG from 200–500 ms; mean baseline, Ridge, linear SVR,
  CatBoost and XGBoost; five shuffled folds, one repetition, fixed parameters.
- Temporal analysis: linear SVR in 10 ms windows, stepping by 10 ms.
- `MODE="quick"`: five folds, one repetition, fixed C=1.
- `MODE="full"`: ten folds repeated ten times; C selected within each training
  fold using three-fold inner validation. This setting affects the temporal
  SVR; the fixed-window benchmark remains the same.
- All presentations of an image are averaged before splitting. Each image
  belongs to only one test fold per repetition. Splits are shared across models.
- EEG uses one RMS scale learned on the training fold for all electrodes,
  preserving relative amplitudes. There is no per-electrode z-score.
  Target ratings are standardized separately inside training folds.
- Pearson r is Fisher-averaged over folds and repetitions; R² and RMSE use
  held-out predictions. Positive r alone does not establish useful prediction.
- Temporal group tests use sign flips of participant Fisher-z correlations
  against zero, with a maximum-cluster correction across the three targets.
  They do not implement the real-minus-shuffled-label control of the final
  classification notebooks.

## Results and scope

`results_quick/` contains completed exploratory results for 35 participants.
The summary and per-participant CSV files use four decimal places.
CatBoost has higher mean r than XGBoost for all three targets in this run,
but its mean RMSE exceeds the mean-prediction baseline. This does not establish
superiority. These participants were already examined in earlier analyses.

Full-mode results are not complete and are not included as final results.
Per-participant joblib caches remain local so interrupted runs can resume.
The cache fingerprint checks code, configuration, EEG metadata and ratings.
Do not alter these inputs during a run if you want to reuse its caches.
