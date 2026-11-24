"""Reproducible food-rating regression with amplitude-preserving normalisation."""
from pathlib import Path
import hashlib
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGETS = {"healthiness": "rating_healthiness", "tastiness": "rating_tastiness",
           "willingness": "rating_willingnessToEat"}


class GlobalScale(TransformerMixin, BaseEstimator):
    """Divide every feature by ONE training-set RMS. Preserve amplitude ratios."""
    def fit(self, X, y=None):
        self.scale_ = float(np.sqrt(np.mean(np.asarray(X, dtype=float) ** 2)))
        if not np.isfinite(self.scale_) or self.scale_ <= 0:
            raise ValueError("Invalid common scale.")
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) / self.scale_


def make_model(name, seed=42):
    if name == "Mean baseline":
        return DummyRegressor()
    if name == "SVR linear":
        model = SVR(kernel="linear", C=1.0, epsilon=0.1)
    elif name == "Ridge":
        model = Ridge(alpha=100.0)
    elif name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=150, depth=3, learning_rate=0.05,
                                  l2_leaf_reg=3, random_seed=seed, thread_count=1,
                                  verbose=False, allow_writing_files=False)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                             reg_lambda=3, objective="reg:squarederror",
                             tree_method="hist", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown model: {name}")
    # y scaling is independent of electrode scaling. Fit both only on training.
    return TransformedTargetRegressor(
        regressor=make_pipeline(GlobalScale(), model), transformer=StandardScaler())


def load_images(data_file, ratings_dir, subject):
    """Read only one subject through xarray; average its three display views."""
    with xr.open_dataset(Path(data_file).expanduser(), engine="netcdf4") as ds:
        subject_ds = ds.sel(recording=subject + "_task-categorisation")
        selected = subject_ds.isel(epoch=np.flatnonzero(subject_ds.event_name.values == "display"))
        codes = selected.food_image_code.values
        unique = np.unique(codes)
        if len(codes) != 360 or len(unique) != 120:
            raise ValueError(f"{subject}: unexpected trial/image count.")
        times = selected["sample"].values * 1000 / float(np.asarray(ds.attrs["sfreq"]).item())
        keep = np.flatnonzero((times >= -100) & (times <= 1000))
        eeg = selected["data"].isel(sample=keep).transpose("epoch", "channel", "sample").values
        if not np.isfinite(eeg).all():
            raise ValueError(f"{subject}: nonfinite EEG.")
        if not np.all(np.unique(codes, return_counts=True)[1] == 3):
            raise ValueError(f"{subject}: expected three views per image.")
        averaged = np.array([eeg[codes == image].mean(axis=0) for image in unique])
    ratings = pd.read_csv(Path(ratings_dir) / f"FOODEEG_p{subject[4:]}_ratings.csv")
    if not ratings.stimCode.is_unique:
        raise ValueError("Duplicate image ratings.")
    ratings.index = "foodImage." + ratings.stimCode.astype(str)
    Y = ratings.loc[unique, list(TARGETS.values())].to_numpy(float)
    if not np.isfinite(Y).all() or np.any(Y.std(axis=0) == 0):
        raise ValueError("Missing or constant ratings.")
    return averaged, Y, unique, times[keep]


def scores(y, prediction, folds):
    z = []
    for k in np.unique(folds):
        a, b = y[folds == k], prediction[folds == k]
        if a.std() < 1e-12:
            raise ValueError("Constant target in a test fold.")
        r = 0.0 if b.std() < 1e-12 else np.corrcoef(a, b)[0, 1]
        z.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    return {"r": float(np.tanh(np.mean(z))),
            "R2": float(r2_score(y, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y, prediction)))}


def evaluate(X, y, name, n_splits, repeats, seed, tune_svr=False):
    """All rows are unique images, so KFold is also image-grouped here."""
    all_scores, predictions = [], []
    started = time.perf_counter()
    for repeat in range(repeats):
        predicted = np.empty(len(y))
        fold_ids = np.empty(len(y), dtype=int)
        splitter = KFold(n_splits, shuffle=True, random_state=seed + repeat)
        for k, (train, test) in enumerate(splitter.split(X)):
            assert set(train).isdisjoint(test)
            model = make_model(name, seed)
            if tune_svr and name == "SVR linear":
                model = GridSearchCV(
                    model, {"regressor__svr__C": [0.001, 0.01, 0.1, 1.0]},
                    cv=KFold(3, shuffle=True, random_state=seed + repeat),
                    scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
            model.fit(X[train], y[train])
            predicted[test] = model.predict(X[test])
            fold_ids[test] = k
        all_scores.append(scores(y, predicted, fold_ids))
        predictions.append(predicted)
    # Fisher aggregation is applied across folds AND repetitions for r.
    r = np.tanh(np.mean(np.arctanh(np.clip([s["r"] for s in all_scores], -0.999999, 0.999999))))
    result = {"r": float(r), "R2": float(np.mean([s["R2"] for s in all_scores])),
              "RMSE": float(np.mean([s["RMSE"] for s in all_scores])),
              "time_s": time.perf_counter() - started}
    return result, np.asarray(predictions)


def run(config, progress=print):
    """Run/resume subject-by-subject, key caches on code, data, ratings and protocol."""
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    subjects = list(config["subjects"])
    data = Path(config["data_file"]).expanduser()
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = {"config": config, "code_sha256": digest,
             "data_size": data.stat().st_size, "data_mtime": data.stat().st_mtime_ns,
             "ratings_sha256": {
                 sub: hashlib.sha256((Path(config["ratings_dir"]) /
                     f"FOODEEG_p{sub[4:]}_ratings.csv").read_bytes()).hexdigest()
                 for sub in subjects}}
    identifier = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    (root / "protocol.json").write_text(json.dumps(stamp, indent=2))
    results = []
    for subject in subjects:
        cache = root / f"{subject}.joblib"
        if cache.exists():
            saved = joblib.load(cache)
            if saved.get("config_id") == identifier:
                results.append(saved)
                progress(f"{subject}: cached")
                continue
        X, Y, images, times = load_images(data, config["ratings_dir"], subject)
        saved = {"config_id": identifier, "subject": subject,
                 "benchmark": [], "temporal": []}
        window = (times >= 200) & (times <= 500)
        features = X[:, :, window].mean(axis=2)
        for j, target in enumerate(TARGETS):
            for model in config["models"]:
                metrics, prediction = evaluate(
                    features, Y[:, j], model, config["benchmark_folds"],
                    config["benchmark_repeats"], config["seed"])
                saved["benchmark"].append(
                    {"subject": subject, "target": target, "model": model, **metrics})
                progress(f"{subject} {target} {model}: r={metrics['r']:.4f}")
        for start in np.arange(-100, 1000, config["step_ms"]):
            mask = (times >= start) & (times < start + config["width_ms"])
            if not mask.any():
                raise ValueError("No EEG samples in a requested temporal window.")
            features = X[:, :, mask].mean(axis=2)
            for j, target in enumerate(TARGETS):
                metrics, _ = evaluate(
                    features, Y[:, j], "SVR linear", config["temporal_folds"],
                    config["temporal_repeats"], config["seed"],
                    tune_svr=config.get("tune_temporal_svr", False))
                saved["temporal"].append(
                    {"subject": subject, "target": target,
                     "time_ms": float(times[mask].mean()),
                     "window_start_ms": float(start), "n_samples": int(mask.sum()), **metrics})
        temporary = cache.with_suffix(".tmp")
        joblib.dump(saved, temporary)
        temporary.replace(cache)
        results.append(saved)
        progress(f"{subject}: complete ({len(results)}/{len(subjects)})")
    detail = pd.DataFrame([row for result in results for row in result["benchmark"]])
    detail.to_csv(root / "benchmark_detail.csv", index=False, float_format="%.4f")
    summary = detail.groupby(["target", "model"]).agg(
        mean_r=("r", "mean"), std_r=("r", "std"),
        n_r_above_0=("r", lambda a: int((a > 0).sum())),
        mean_R2=("R2", "mean"), mean_RMSE=("RMSE", "mean")).reset_index()
    summary.to_csv(root / "benchmark_summary.csv", index=False, float_format="%.4f")
    temporal = pd.DataFrame([row for result in results for row in result["temporal"]])
    temporal.to_csv(root / "temporal_detail.csv", index=False, float_format="%.4f")
    return detail, summary, temporal


def report(temporal, output_dir, n_permutations=5000, seed=42):
    """Exploratory one-sided Fisher-z group sign-flip clusters, all three targets."""
    root = Path(output_dir)
    subjects = sorted(temporal.subject.unique())
    times = np.sort(temporal.time_ms.unique())
    z = []
    for target in TARGETS:
        matrix = temporal[temporal.target == target].pivot(
            index="subject", columns="time_ms", values="r").loc[subjects, times].to_numpy()
        if not np.isfinite(matrix).all():
            raise ValueError("Incomplete temporal results.")
        z.append(np.arctanh(np.clip(matrix, -0.999999, 0.999999)))
    z = np.stack(z, axis=1)
    n = len(subjects)
    if n < 3:
        raise ValueError("At least three subjects are needed for group reporting.")
    def t_stat(values):
        return values.mean(axis=0) / np.maximum(values.std(axis=0, ddof=1)/np.sqrt(n), 1e-12)
    threshold = stats.t.ppf(0.95, n-1)
    def clusters(curve):
        labels, count = ndimage.label(curve > threshold)
        return [(np.flatnonzero(labels == i), curve[labels == i].sum())
                for i in range(1, count+1)]
    observed = t_stat(z)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        t = t_stat(z * rng.choice([-1, 1], (n, 1, 1)))
        null.append(max([mass for curve in t for _,mass in clusters(curve)], default=0))
    rows = []
    for i,target in enumerate(TARGETS):
        for idx,mass in clusters(observed[i]):
            rows.append({"target":target, "start_ms":times[idx[0]], "end_ms":times[idx[-1]],
                         "p_fwer":(1+np.sum(np.asarray(null)>=mass))/(n_permutations+1)})
    cluster_table = pd.DataFrame(rows,columns=["target","start_ms","end_ms","p_fwer"])
    cluster_table.to_csv(root/"temporal_clusters.csv",index=False,float_format="%.4f")
    fig, axes = plt.subplots(1,3,figsize=(14,4),sharey=True)
    for i,(ax,target) in enumerate(zip(axes,TARGETS)):
        individual = np.tanh(z[:,i])
        for curve in individual:ax.plot(times,curve,color="0.85",linewidth=0.5)
        ax.plot(times,individual.mean(axis=0),color="tab:blue",label="Mean")
        ax.axhline(0,color="0.3",linestyle="--")
        ax.axvline(0,color="0.3",linestyle=":")
        ax.set(title=target,xlabel="Time from image onset (ms)",ylabel="Pearson r")
        ax.legend()
    fig.tight_layout()
    fig.savefig(root/"temporal_svr.png",dpi=170)
    return cluster_table,fig
