from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycaret.classification import *

# --------------------------------------------------
# Rutas
# --------------------------------------------------
try:
    CARPETA = Path(__file__).resolve().parent
except NameError:
    CARPETA = Path.cwd()

CARPETA_PNG = CARPETA / "png"
CARPETA_PNG.mkdir(exist_ok=True)

# --------------------------------------------------
# Datos y configuración
# --------------------------------------------------
df = pd.read_csv("dataset/Covid_Dataset_Limpio_V4_balanceado.csv")

enf = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC"
]

cat = [
    "USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE",
    "INTUBED", "PNEUMONIA", "PREGNANT", *enf,
    "TOBACCO", "CLASIFFICATION_FINAL", "ICU"
]

setup(
    data=df,
    target="MUERTO",
    train_size=0.8,
    fold=10,
    session_id=42,
    ignore_features=["DATE_DIED"],
    categorical_features=cat,
    fix_imbalance=False,
    verbose=False,
)

# --------------------------------------------------
# Modelos y métricas
# --------------------------------------------------
ids = ["catboost", "lightgbm", "rf", "xgboost", "gbc", "lr"]

METRICAS = {
    "Accuracy": "Accuracy",
    "Prec.": "Precision",
    "Recall": "Recall",
    "F1": "F1",
    "AUC": "AUC",
}

# --------------------------------------------------
# Comparación general
# --------------------------------------------------
compare_models(
    include=ids,
    sort="F1",
    n_select=3,
    verbose=False,
)

print("\n=== COMPARACIÓN GENERAL DE MODELOS ===")
print(pull().to_string())

# --------------------------------------------------
# Entrenamiento: guardar folds y medias en una pasada
# --------------------------------------------------
folds = {}
medias = {}
desviaciones = {}

for id_ in ids:
    print(f"\nEntrenando {id_}...", flush=True)

    create_model(id_, fold=10, verbose=False)
    tabla = pull()

    # Mostrar los 10 folds por consola
    resultados = tabla.iloc[:10][list(METRICAS.keys())].astype(float)
    resultados.index = [f"Fold {i}" for i in range(1, 11)]

    print(f"\n=== {id_.upper()}: RESULTADOS POR FOLD ===")
    print(resultados.round(4).to_string())

    # Guardar valores individuales para los gráficos
    folds[id_] = resultados.copy()

    # Guardar estadísticas calculadas por PyCaret
    medias[id_] = tabla.loc["Mean", list(METRICAS.keys())].astype(float)
    desviaciones[id_] = tabla.loc["Std", list(METRICAS.keys())].astype(float)

    resumen = pd.DataFrame({
        "Media": medias[id_],
        "Desviación estándar": desviaciones[id_],
    })

    print(f"\n=== {id_.upper()}: RESUMEN ===")
    print(resumen.round(4).to_string())

# --------------------------------------------------
# Resumen general de medias
# --------------------------------------------------
resumen_general = pd.DataFrame(medias).T
resumen_general.index.name = "Modelo"
resumen_general = resumen_general.sort_values("F1", ascending=False)

print("\n=== RESUMEN GENERAL: MEDIAS DE LOS 10 FOLDS ===")
print(resumen_general.round(4).to_string())

# --------------------------------------------------
# Gráficos lineales por fold
# --------------------------------------------------
DESTACAR = "lightgbm"
APARTE = ["rf"]

MARCADORES_ALG = dict(zip(ids, ["o", "s", "^", "D", "v", "P"]))
COLORES_ALG = {k: f"C{i}" for i, k in enumerate(ids)}
x_folds = np.arange(1, 11)


def valores_fold(k, col):
    return folds[k][col].astype(float).to_numpy()


def media(k, col):
    return float(medias[k][col])


for col, nombre in METRICAS.items():
    orden = sorted(ids, key=lambda k: media(k, col), reverse=True)

    arriba = [k for k in orden if k not in APARTE]
    abajo = [k for k in orden if k in APARTE]

    def rango(keys):
        return (
            min(valores_fold(k, col).min() for k in keys),
            max(valores_fold(k, col).max() for k in keys),
        )

    (a0, a1), (b0, b1) = rango(arriba), rango(abajo)

    span = max(a1 - a0, b1 - b0) * 1.15
    if span == 0:
        span = 0.01

    def limites(lo, hi):
        centro = (lo + hi) / 2
        return centro - span / 2, centro + span / 2

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(13, 7.5),
        gridspec_kw={"hspace": 0.08},
    )

    lineas = {}
    etiquetas = {}

    for ax, keys in ((ax1, arriba), (ax2, abajo)):
        for f in x_folds[::2]:
            ax.axvspan(
                f - 0.5, f + 0.5,
                color="gray", alpha=0.08, zorder=0,
            )

        for k in keys:
            destacado = k == DESTACAR

            linea, = ax.plot(
                x_folds,
                valores_fold(k, col),
                marker=MARCADORES_ALG[k],
                color=COLORES_ALG[k],
                linewidth=3.0 if destacado else 1.6,
                markersize=9 if destacado else 6,
                alpha=1.0 if destacado else 0.85,
                markeredgecolor="black",
                markeredgewidth=0.4,
                zorder=4 if destacado else 3,
            )

            lineas[k] = linea
            etiquetas[k] = f"{k} (media {media(k, col):.4f})"

        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0.5, 10.5)

    ax1.set_ylim(*limites(a0, a1))
    ax2.set_ylim(*limites(b0, b1))

    # Eje discontinuo entre paneles
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(bottom=False)

    d = 0.012
    kw = dict(color="k", clip_on=False, linewidth=1)

    ax1.plot((-d, d), (-d, d), transform=ax1.transAxes, **kw)
    ax1.plot((1 - d, 1 + d), (-d, d), transform=ax1.transAxes, **kw)
    ax2.plot((-d, d), (1 - d, 1 + d), transform=ax2.transAxes, **kw)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kw)

    ax2.set_xlabel("Fold", fontsize=11)
    ax2.set_xticks(x_folds)
    fig.supylabel(nombre, fontsize=12)

    brechas = ", ".join(
        f"{k}: {media(DESTACAR, col) - media(k, col):.4f} por debajo"
        for k in abajo
    )

    fig.suptitle(
        f"{nombre} por fold (validación cruzada de 10 folds)\n"
        f"Panel inferior = {', '.join(abajo)} ({brechas} de {DESTACAR}). "
        "Misma escala en ambos paneles.",
        fontsize=11,
        y=0.98,
    )

    fig.legend(
        [lineas[k] for k in orden],
        [etiquetas[k] for k in orden],
        title="Algoritmo (ordenado por media)",
        loc="center left",
        bbox_to_anchor=(0.76, 0.50),
        fontsize=9,
        title_fontsize=9,
        frameon=True,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.73,
        top=0.84,
        bottom=0.10,
    )

    fig.savefig(
        CARPETA_PNG / f"lineas_{nombre.lower().replace('.', '')}.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

print(f"\nGráficos guardados en: {CARPETA_PNG}")

