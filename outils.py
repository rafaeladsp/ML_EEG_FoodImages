"""Loading, validation and plotting for per-subject "Healthy?" decoding."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

DATA_FILE = Path(__file__).with_name("epoch_data_6sujets.nc")


def load_subject(name, data_file=None):
    """Return X (epoch, channel, time_ms), y (No=0, Yes=1) and the image codes.

    Labels are taken from the "health" trials and copied to every "display"
    trial of the same image. Only the requested subject is read.
    Without an explicit path, look for the file next to this module, then in Downloads.
    """
    if data_file is None:
        data_file = DATA_FILE if DATA_FILE.is_file() else Path.home() / "Downloads" / DATA_FILE.name
    recording = name if name.endswith("_task-categorisation") else f"{name}_task-categorisation"
    with xr.open_dataset(Path(data_file).expanduser(), engine="netcdf4") as ds:
        if recording not in ds.indexes["recording"]:
            raise ValueError(f"Recording not found: {recording}")
        subject = ds.sel(recording=recording)
        epochs = np.flatnonzero(subject["event_name"].values == "display")
        trials = subject.isel(epoch=epochs)
        images = trials["food_image_code"].values
        health = trials["event_type"].values == "health"
        answers = trials["resp_choice"].values[health]

        health_images = images[health]
        if np.any(np.isin(images, ["", "nan", "None"])):
            raise ValueError(f"{name}: missing image code.")
        if len(np.unique(health_images)) != len(health_images):
            raise ValueError(f"{name}: several Healthy? trials for one image.")
        if not np.isin(answers, ["Yes", "No"]).all():
            raise ValueError(f"{name}: missing or invalid Healthy? answer.")
        label = dict(zip(health_images, answers == "Yes"))
        missing = set(images) - label.keys()
        if missing:
            raise ValueError(f"{name}: {len(missing)} images without a Healthy? answer.")
        y = np.array([label[image] for image in images], dtype=int)

        time_ms = subject["sample"].values * 1000 / np.asarray(ds.attrs["sfreq"]).item()
        window = np.flatnonzero((time_ms >= -100) & (time_ms <= 1000))
        start, stop = window[0], window[-1] + 1
        X = (
            trials["data"]
            .isel(sample=slice(start, stop))
            .reset_coords(drop=True)
            .rename({"sample": "time_ms"})
            .assign_coords(
                time_ms=time_ms[start:stop],
                food_image_code=("epoch", images),
            )
            .rename("EEG")
            .load()
        )
    return X, y, images


def decode_time(X, y, groups, clf, *, n_splits=5, seed=42, grouped=True):
    """Return the mean AUC at each time point, using the same folds at every time point.

    Pass grouped=False to compare with a per-trial stratified cross-validation.
    Pass a Pipeline if the classifier needs standardisation or PCA.
    """
    eeg = X.transpose("epoch", "channel", "time_ms").values
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.shape != (len(eeg),) or groups.shape != y.shape:
        raise ValueError("X, y and groups must match trial by trial.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Binary AUC requires both classes 0 and 1.")

    if grouped:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(cv.split(eeg[:, :, 0], y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(cv.split(eeg[:, :, 0], y))
    if any(len(np.unique(y[idx])) < 2 for fold in folds for idx in fold):
        raise ValueError("A fold contains only one class. Reduce n_splits.")

    auc = [
        cross_val_score(clf, eeg[:, :, t], y, cv=folds, scoring="roc_auc", error_score="raise").mean()
        for t in range(eeg.shape[2])
    ]
    return xr.DataArray(auc, dims="time_ms", coords={"time_ms": X.time_ms}, name="auc")


def chance_control(X, y, groups, clf, *, n_permutations=20, n_splits=5, seed=42):
    """Return one AUC curve per permutation of the labels across images."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    y = np.asarray(y)
    image_labels = y[first]
    if not np.array_equal(y, image_labels[inverse]):
        raise ValueError("Each image must have a single label.")

    rng = np.random.default_rng(seed)
    curves = [
        decode_time(
            X, rng.permutation(image_labels)[inverse], groups, clf,
            n_splits=n_splits, seed=seed,
        )
        for _ in range(n_permutations)
    ]
    return xr.concat(curves, dim="permutation", join="exact")


def plot_curves(curves):
    """Plot a list of individual curves and return the figure and its axes."""
    auc = xr.concat(list(curves), dim="subject", join="exact")
    fig, ax = plt.subplots()
    for i, curve in enumerate(auc):
        ax.plot(auc.time_ms, curve, color="0.7", label="Subjects" if i == 0 else None)
    ax.plot(auc.time_ms, auc.mean("subject"), color="tab:blue", linewidth=2, label="Mean")
    ax.axhline(0.5, color="0.35", linestyle="--", label="Chance")
    ax.axvline(0, color="0.35", linestyle=":")
    ax.set(xlabel="Time from image onset (ms)", ylabel="ROC AUC")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# French names kept as aliases, so that existing notebooks still run.
charger_sujet = load_subject
decoder_temps = decode_time
controle_hasard = chance_control
tracer = plot_curves