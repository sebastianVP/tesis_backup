"""Figura 3.6 · Comparación en prueba sobre los mismos 159 367 instantes (B3).
Valores de la Tabla 3.8 (V08_metricas_prueba.csv y V08b_referencias_val_test.csv; v2.0: tabla_final_test_comun.csv).
Si se ejecuta desde la raíz del proyecto y existe validacion/salidas/V08_metricas_prueba.csv, comprueba los valores del v3.0,
la variante B y las referencias contra ese archivo.
Uso: python fig3_6_prueba.py  →  Figura_3_6_prueba_v6_1.png y .pdf"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = {  # modelo: (HSS, POD, Precisión, inicios por ventana, RMSE en eventos)
    "v3.0 (principal)":          (0.5451, 0.4372, 0.7406, 0.3254, 0.2225),
    "Variante B":                (0.4984, 0.3691, 0.7888, 0.2506, 0.2723),
    "v2.0 (anterior)":           (0.451, 0.32, 0.77, 0.21, 0.249),
    "Persistencia (0.6)":        (0.3207, 0.1968, 0.9170, 0.0000, 0.3041),
    "Regla de S4 actual (0.40)": (0.5857, 0.5093, 0.7027, 0.3891, 0.3041),
    "AR(20) (0.40)":             (0.5876, 0.4662, 0.8104, 0.3409, 0.3208),
}
f = Path("validacion/salidas/V08_metricas_prueba.csv")
if f.exists():
    import pandas as pd
    d = pd.read_csv(f); d = d[d["mascara"].str.startswith("tesis")].set_index("modelo")
    mapa = {"v3.0 (principal)": "v3.0", "Variante B": "variante B", "Persistencia (0.6)": "Persistencia (0.6)",
            "Regla de S4 actual (0.40)": "Persistencia (umbral calibrado)", "AR(20) (0.40)": "AR(20) multipaso"}
    for k, v in mapa.items():
        r = d.loc[v]; ref = (r["HSS"], r["POD"], r["Precision"], r["inicios_por_ventana"], r["RMSE_evento"])
        assert np.allclose(ref, M[k], atol=6e-4), f"{k}: no coincide con V08"
    print("Comprobado contra V08_metricas_prueba.csv")

plt.rcParams.update({"figure.dpi": 300, "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = ["#1F4E79", "#4A9BE8", "#B5791F", "#8C8C8C", "#2E8B57", "#8E44AD"]
nombres = list(M); vals = np.array([M[k] for k in nombres])
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [3.2, 1.2]})
met = ["HSS", "Detección (POD)", "Precisión", "Inicios anticipados\n(por ventana)"]
x = np.arange(len(met)); w = 0.135
for i, (n, c) in enumerate(zip(nombres, COL)):
    barras = a.bar(x + (i - 2.5) * w, vals[i, :4], w, color=c, label=n, edgecolor="white", lw=0.5,
                   hatch="//" if "0.40" in n else None)
    for xb, v in zip(x + (i - 2.5) * w, vals[i, :4]):
        a.text(xb, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, rotation=90)
a.set_xticks(x); a.set_xticklabels(met); a.set_ylim(0, 1.08); a.set_ylabel("Valor (0–1)")
a.set_title("a) Habilidad de alerta", fontsize=11); a.grid(axis="y", alpha=0.3)
a.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
orden = np.argsort(vals[:, 4])
b.barh(np.arange(len(nombres)), vals[orden, 4], color=[COL[i] for i in orden], hatch=None)
for j, i in enumerate(orden):
    b.text(vals[i, 4] + 0.004, j, f"{vals[i, 4]:.3f}", va="center", fontsize=8)
b.set_yticks(np.arange(len(nombres))); b.set_yticklabels([nombres[i] for i in orden], fontsize=8.5)
b.set_xlim(0, 0.38); b.set_xlabel("RMSE en eventos (S4 ≥ 0.6)"); b.set_title("b) Error en eventos (menor es mejor)", fontsize=11)
b.grid(axis="x", alpha=0.3)
fig.text(0.5, -0.03, "Rayado: referencias con umbral calibrado en validación (re-análisis de la auditoría). Mismos 159 367 instantes de prueba.",
         ha="center", fontsize=8.5, style="italic", color="#444444")
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"Figura_3_6_prueba_v6_1.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("Figura_3_6_prueba_v6_1.png y .pdf generadas")
