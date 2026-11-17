"""Chargement, validation et visualisation du décodage Healthy? par sujet."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)

FICHIER = Path(__file__).with_name("epoch_data_6sujets.nc")


def charger_sujet(nom, fichier=None):
    """Renvoie X (epoch, channel, time_ms), y (No=0, Yes=1) et les codes image.

    Les étiquettes sont tirées des essais health et reportées sur tous les
    essais display de la même image. Seul le sujet demandé est lu.
    Sans chemin explicite, cherche le fichier près du module puis dans Downloads.
    """
    if fichier is None:
        fichier = FICHIER if FICHIER.is_file() else Path.home() / "Downloads" / FICHIER.name
    recording = nom if nom.endswith("_task-categorisation") else f"{nom}_task-categorisation"
    with xr.open_dataset(Path(fichier).expanduser(), engine="netcdf4") as ds:
        if recording not in ds.indexes["recording"]:
            raise ValueError(f"Enregistrement introuvable : {recording}")
        sujet = ds.sel(recording=recording)
        epochs = np.flatnonzero(sujet["event_name"].values == "display")
        essais = sujet.isel(epoch=epochs)
        images = essais["food_image_code"].values
        health = essais["event_type"].values == "health"
        reponses = essais["resp_choice"].values[health]

        images_health = images[health]
        if np.any(np.isin(images, ["", "nan", "None"])):
            raise ValueError(f"{nom} : code image manquant.")
        if len(np.unique(images_health)) != len(images_health):
            raise ValueError(f"{nom} : plusieurs essais Healthy? pour une image.")
        if not np.isin(reponses, ["Yes", "No"]).all():
            raise ValueError(f"{nom} : réponse Healthy? manquante ou invalide.")
        etiquette = dict(zip(images_health, reponses == "Yes"))
        manquantes = set(images) - etiquette.keys()
        if manquantes:
            raise ValueError(f"{nom} : {len(manquantes)} images sans réponse Healthy?.")
        y = np.array([etiquette[image] for image in images], dtype=int)

        temps = sujet["sample"].values * 1000 / np.asarray(ds.attrs["sfreq"]).item()
        fenetre = np.flatnonzero((temps >= -100) & (temps <= 1000))
        debut, fin = fenetre[0], fenetre[-1] + 1
        X = (
            essais["data"]
            .isel(sample=slice(debut, fin))
            .reset_coords(drop=True)
            .rename({"sample": "time_ms"})
            .assign_coords(
                time_ms=temps[debut:fin],
                food_image_code=("epoch", images),
            )
            .rename("EEG")
            .load()
        )
    return X, y, images


def decoder_temps(X, y, groupes, clf, *, n_splits=5, seed=42, grouper=True):
    """Renvoie l'AUC moyenne par instant avec les mêmes plis à chaque instant.

    Passer grouper=False pour comparer à une validation stratifiée par essai.
    Fournir un Pipeline si le classifieur nécessite une standardisation ou PCA.
    """
    eeg = X.transpose("epoch", "channel", "time_ms").values
    y = np.asarray(y)
    groupes = np.asarray(groupes)
    if y.shape != (len(eeg),) or groupes.shape != y.shape:
        raise ValueError("X, y et groupes doivent correspondre essai par essai.")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("L'AUC binaire requiert les deux classes 0 et 1.")

    if grouper:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        plis = list(cv.split(eeg[:, :, 0], y, groupes))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        plis = list(cv.split(eeg[:, :, 0], y))
    if any(len(np.unique(y[idx])) < 2 for pli in plis for idx in pli):
        raise ValueError("Un pli ne contient qu'une classe. Réduire n_splits.")

    auc = [
        cross_val_score(clf, eeg[:, :, t], y, cv=plis, scoring="roc_auc", error_score="raise").mean()
        for t in range(eeg.shape[2])
    ]
    return xr.DataArray(auc, dims="time_ms", coords={"time_ms": X.time_ms}, name="auc")


def controle_hasard(X, y, groupes, clf, *, n_permutations=20, n_splits=5, seed=42):
    """Renvoie une courbe d'AUC par permutation des étiquettes entre images."""
    if n_permutations < 1:
        raise ValueError("n_permutations doit être positif.")
    _, premieres, inverse = np.unique(groupes, return_index=True, return_inverse=True)
    y = np.asarray(y)
    labels_image = y[premieres]
    if not np.array_equal(y, labels_image[inverse]):
        raise ValueError("Chaque image doit avoir une seule étiquette.")

    rng = np.random.default_rng(seed)
    courbes = [
        decoder_temps(
            X, rng.permutation(labels_image)[inverse], groupes, clf,
            n_splits=n_splits, seed=seed,
        )
        for _ in range(n_permutations)
    ]
    return xr.concat(courbes, dim="permutation", join="exact")


def tracer(courbes):
    """Trace une liste de courbes individuelles et renvoie la figure et ses axes."""
    auc = xr.concat(list(courbes), dim="sujet", join="exact")
    fig, ax = plt.subplots()
    for i, courbe in enumerate(auc):
        ax.plot(auc.time_ms, courbe, color="0.7", label="Sujets" if i == 0 else None)
    ax.plot(auc.time_ms, auc.mean("sujet"), color="tab:blue", linewidth=2, label="Moyenne")
    ax.axhline(0.5, color="0.35", linestyle="--", label="Hasard")
    ax.axvline(0, color="0.35", linestyle=":")
    ax.set(xlabel="Temps depuis l’apparition de l’image (ms)", ylabel="AUC ROC")
    ax.legend()
    fig.tight_layout()
    return fig, ax
