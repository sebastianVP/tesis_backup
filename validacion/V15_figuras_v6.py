"""V15 · Figuras 3.1 a 3.5 de la tesis con los resultados de la repetición v6.

Lee experiments/registro_tesis_v6.csv (validación) y, para las Figuras 3.4b y 3.5b,
reports/tables/V6_E5_latencia.csv y reports/tables/V6_E6_importancia_E.csv.
Guarda PNG de 300 ppp en reports/figures/v6/ y una tabla con los valores graficados.

Uso (desde la raíz del proyecto):
    python validacion/V15_figuras_v6.py
    python validacion/V15_figuras_v6.py --registro experiments/registro_tesis_v6.csv --salida reports/figures/v6
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--registro", default="experiments/registro_tesis_v6.csv")
ap.add_argument("--tablas", default="reports/tables")
ap.add_argument("--salida", default="reports/figures/v6")
ap.add_argument("--dpi", type=int, default=300)
a = ap.parse_args()

SAL = Path(a.salida); SAL.mkdir(parents=True, exist_ok=True)
R = pd.read_csv(a.registro)
R = R[R["particion"] == "val"].copy()
print(f"Registro: {a.registro} · {len(R)} filas de validación · grupos: {sorted(R['grupo'].unique())}")

AZUL, CELESTE, NARANJA, GRIS, ROJO, VERDE = "#1f4e79", "#5b9bd5", "#c55a11", "#7f7f7f", "#c00000", "#548235"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.frameon": False, "figure.dpi": 100})
VALORES = []

def grupo(g):
    x = R[R["grupo"] == g]
    if x.empty:
        sys.exit(f"ERROR: el grupo {g} no está en {a.registro}")
    return x

def resumen(x, met):
    t = x.groupby("modelo", sort=False)[met].agg(["mean", "std", "count"]).reset_index()
    t["std"] = t["std"].fillna(0.0)
    return t

def anotar(fig_id, t, met):
    for _, f in t.iterrows():
        VALORES.append(dict(figura=fig_id, modelo=f["modelo"], metrica=met, media=round(f["mean"], 4),
                            desviacion=round(f["std"], 4), semillas=int(f["count"])))

def guardar(fig, nombre):
    p = SAL / nombre
    fig.savefig(p, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  ✓ {p}")

# ---------------------------------------------------------------- Figura 3.1 · ventana (E2)
x = grupo("E2_ventana").copy()
x["L"] = x["modelo"].str.extract(r"LB(\d+)").astype(int)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, (met, nom, col) in enumerate((("HSS", "HSS (umbral 0.6)", AZUL), ("RMSE_evento", "RMSE en eventos", NARANJA))):
    t = x.groupby("L")[met].agg(["mean", "std"]).reset_index().sort_values("L"); t["std"] = t["std"].fillna(0)
    ax[k].errorbar(t["L"], t["mean"], yerr=t["std"], marker="o", color=col, capsize=4, lw=1.8)
    ax[k].set_xscale("log"); ax[k].set_xticks(t["L"]); ax[k].set_xticklabels(t["L"])
    ax[k].set_xlabel("Ventana de historia L (min)"); ax[k].set_ylabel(nom)
    ax[k].set_title(f"{'a' if k == 0 else 'b'}) {nom}")
    for _, f in t.iterrows():
        VALORES.append(dict(figura="3.1", modelo=f"L={int(f['L'])}", metrica=met, media=round(f["mean"], 4), desviacion=round(f["std"], 4), semillas=3))
fig.suptitle("Sensibilidad a la ventana de historia (validación, 3 semillas, β = 100)", y=1.02)
guardar(fig, "fig3_1_ventana_v6.png")

# ---------------------------------------------------------------- Figura 3.2 · ablación (E3 + A')
x = pd.concat([grupo("E3_ablacion"), R[(R["grupo"] == "E3b_ablacion_extra") & R["modelo"].str.startswith("A'")]])
orden = ["A · S4 + MSE", "B · + hora + MSE", "A' · S4 + WFL", "C · + hora + WFL", "D · + TEC/ROTEC/ROTI", "D' · + ROTI", "E · + índices globales"]
etiq = ["A · S4\nMSE", "B · S4+hora\nMSE", "A' · S4\nWFL", "C · S4+hora\nWFL", "D · +TEC\nROTEC ROTI", "D' · +ROTI", "E · +índices"]
t1, t2 = resumen(x, "HSS"), resumen(x, "HSS_opt")
t1 = t1.set_index("modelo").reindex(orden); t2 = t2.set_index("modelo").reindex(orden)
pers = x.loc[x["modelo"] == "Persistencia", "HSS"]
fig, ax = plt.subplots(figsize=(10.5, 4.2)); i = np.arange(len(orden)); w = 0.38
ax.bar(i - w / 2, t1["mean"], w, yerr=t1["std"], capsize=3, color=AZUL, label="Umbral fijo 0.6")
ax.bar(i + w / 2, t2["mean"], w, yerr=t2["std"], capsize=3, color=CELESTE, label="Umbral calibrado en validación")
if len(pers):
    ax.axhline(pers.mean(), ls="--", color=GRIS, lw=1.2, label=f"Persistencia (0.6): {pers.mean():.3f}")
for j, (v1, s1, v2, s2) in enumerate(zip(t1["mean"], t1["std"], t2["mean"], t2["std"])):
    ax.text(j - w / 2, v1 + s1 + 0.012, f"{v1:.2f}", ha="center", fontsize=8)
    if not np.isnan(v2): ax.text(j + w / 2, v2 + s2 + 0.012, f"{v2:.2f}", ha="center", fontsize=8)
ax.set_xticks(i); ax.set_xticklabels(etiq, fontsize=8.5); ax.set_ylabel("HSS en validación"); ax.set_ylim(0, 0.9)
ax.set_title("Estudio de ablación (validación, L = 60 min, β = 100, media ± desviación de 3 semillas)")
ax.legend(loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
anotar("3.2", t1.reset_index(), "HSS"); anotar("3.2", t2.reset_index(), "HSS_opt")
guardar(fig, "fig3_2_ablacion_v6.png")

# ---------------------------------------------------------------- Figura 3.3 · α y β (E4)
f1 = grupo("E4_fase1").copy()
f1["alpha"] = f1["modelo"].str.extract(r"a([\d.]+)_").astype(float); f1["beta"] = f1["modelo"].str.extract(r"_b(\d+)").astype(int)
f2 = grupo("E4_fase2").copy(); f2["beta"] = f2["modelo"].str.extract(r"b(\d+)").astype(int)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for al, col in zip(sorted(f1["alpha"].unique()), (CELESTE, AZUL, NARANJA)):
    t = f1[f1["alpha"] == al].groupby("beta")["HSS"].mean().sort_index()
    ax[0].plot(t.index, t.values, marker="o", color=col, label=f"α = {al:.1f}")
    for b, v in t.items(): VALORES.append(dict(figura="3.3a", modelo=f"a{al}_b{b}", metrica="HSS", media=round(v, 4), desviacion=0.0, semillas=1))
ax[0].set_xscale("log"); ax[0].set_xticks(sorted(f1["beta"].unique())); ax[0].set_xticklabels(sorted(f1["beta"].unique()))
ax[0].set_xlabel("β"); ax[0].set_ylabel("HSS (umbral 0.6)"); ax[0].set_title("a) Fase 1: umbral fijo (1 semilla)"); ax[0].legend(fontsize=8.5)
t = f2.groupby("beta").agg(m=("HSS_opt", "mean"), s=("HSS_opt", "std"), u=("umbral_opt", "mean")).reset_index().sort_values("beta"); t["s"] = t["s"].fillna(0)
ax[1].errorbar(t["beta"], t["m"], yerr=t["s"], marker="o", color=AZUL, capsize=4, lw=1.8)
for _, f in t.iterrows():
    ax[1].annotate(f"u = {f['u']:.2f}", (f["beta"], f["m"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    VALORES.append(dict(figura="3.3b", modelo=f"b{int(f['beta'])}", metrica="HSS_opt", media=round(f["m"], 4), desviacion=round(f["s"], 4), semillas=3))
ax[1].axvspan(5, 25, color=VERDE, alpha=0.08, label="Meseta β = 5–25")
ax[1].set_xscale("log"); ax[1].set_xticks(t["beta"]); ax[1].set_xticklabels(t["beta"])
ax[1].set_xlabel("β (α = 1.5)"); ax[1].set_ylabel("HSS calibrado"); ax[1].set_title("b) Fase 2: umbral calibrado (3 semillas)"); ax[1].legend(fontsize=8.5, loc="lower left")
fig.suptitle("Sensibilidad a α y β en validación (S4 + hora + ROTI, L = 180 min)", y=1.02)
guardar(fig, "fig3_3_alpha_beta_v6.png")

# ---------------------------------------------------------------- Figura 3.4 · LSTM frente a PatchTST (E5)
x = grupo("E5_patchtst"); orden = ["LSTM apilada", "PatchTST", "PatchTST sin RevIN"]
t = resumen(x, "HSS_opt").set_index("modelo").reindex(orden)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
cols = (AZUL, NARANJA, ROJO)
ax[0].bar(orden, t["mean"], yerr=t["std"], capsize=4, color=cols)
for j, m in enumerate(orden):
    sem = x[x["modelo"] == m]["HSS_opt"].to_numpy()
    ax[0].scatter(np.full(len(sem), j), sem, color="black", s=14, zorder=3)
    ax[0].text(j, t.loc[m, "mean"] / 2, f"{t.loc[m, 'mean']:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax[0].set_ylabel("HSS calibrado (validación)"); ax[0].set_ylim(0, 0.8); ax[0].set_title("a) Habilidad de alerta (3 semillas)")
anotar("3.4a", t.reset_index(), "HSS_opt")
lat = Path(a.tablas) / "V6_E5_latencia.csv"
if lat.exists():
    L = pd.read_csv(lat); L = L[L["dispositivo"].str.contains("CPU")].set_index("modelo").reindex(orden)
    ax[1].bar(orden, L["lat_ms_mediana"], color=cols)
    for j, m in enumerate(orden):
        ax[1].text(j, L.loc[m, "lat_ms_mediana"] + 0.4, f"{L.loc[m, 'lat_ms_mediana']:.1f} ms\n{int(L.loc[m, 'parametros']):,} parám.".replace(",", " "), ha="center", fontsize=8)
        VALORES.append(dict(figura="3.4b", modelo=m, metrica="latencia_CPU_ms", media=round(L.loc[m, "lat_ms_mediana"], 2), desviacion=0.0, semillas=1))
    ax[1].set_ylabel("Latencia mediana en CPU (ms)"); ax[1].set_ylim(0, L["lat_ms_mediana"].max() * 1.35); ax[1].set_title("b) Costo computacional")
else:
    ax[1].text(0.5, 0.5, f"No se encontró\n{lat}", ha="center", va="center", transform=ax[1].transAxes); ax[1].set_axis_off()
for k in (0, 1): ax[k].tick_params(axis="x", labelsize=8.5)
fig.suptitle("LSTM apilada frente a PatchTST (validación, S4 + hora + ROTI, L = 180 min, α = 1.5, β = 10)", y=1.02)
guardar(fig, "fig3_4_patchtst_v6.png")

# ---------------------------------------------------------------- Figura 3.5 · precursores (E6)
x = grupo("E6_precursores").copy()
x["L"] = x["modelo"].str.extract(r"LB(\d+)").astype(int)
x["variante"] = x["modelo"].str.replace(r"\s*·\s*LB\d+$", "", regex=True)
var = ["C · S4 + hora", "D' · + ROTI", "D · + TEC/ROTEC/ROTI", "E · completo"]
eti = {"C · S4 + hora": "S4 + hora", "D' · + ROTI": "+ ROTI", "D · + TEC/ROTEC/ROTI": "+ TEC, ROTEC, ROTI", "E · completo": "+ todos"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); i = np.arange(len(var)); w = 0.38
for k, (Lv, col) in enumerate(((60, CELESTE), (180, AZUL))):
    t = x[x["L"] == Lv].groupby("variante")["HSS_opt"].agg(["mean", "std"]).reindex(var); t["std"] = t["std"].fillna(0)
    ax[0].bar(i + (k - 0.5) * w, t["mean"], w, yerr=t["std"], capsize=3, color=col, label=f"L = {Lv} min")
    for j, v in enumerate(var):
        if not np.isnan(t.loc[v, "mean"]):
            VALORES.append(dict(figura="3.5a", modelo=f"{v} · LB{Lv}", metrica="HSS_opt", media=round(t.loc[v, "mean"], 4), desviacion=round(t.loc[v, "std"], 4), semillas=3))
ref = x[x["variante"] == "C · S4 + hora"]["HSS_opt"].mean()
ax[0].axhspan(ref - 0.03, ref + 0.03, color=GRIS, alpha=0.12, label="S4 + hora ± 0.03 (regla)")
ax[0].set_xticks(i); ax[0].set_xticklabels([eti[v] for v in var], fontsize=8.5); ax[0].set_ylim(0.5, 0.75)
ax[0].set_ylabel("HSS calibrado (validación)"); ax[0].set_title("a) Precursores con datos corregidos"); ax[0].legend(fontsize=8, loc="upper left")
imp = Path(a.tablas) / "V6_E6_importancia_E.csv"
if imp.exists():
    I = pd.read_csv(imp).sort_values("caida_HSS_mean")
    ax[1].barh(I["grupo"], I["caida_HSS_mean"], xerr=I["caida_HSS_std"], capsize=3, color=[ROJO if g == "ROTI" else AZUL for g in I["grupo"]])
    xmax = float((I["caida_HSS_mean"] + I["caida_HSS_std"]).max())
    for _, f in I.iterrows():
        ax[1].text(xmax * 1.6, f["grupo"], f"{f['caida_HSS_mean']:.4f}", va="center", ha="left", fontsize=8)
        VALORES.append(dict(figura="3.5b", modelo=f["grupo"], metrica="caida_HSS", media=round(f["caida_HSS_mean"], 4), desviacion=round(f["caida_HSS_std"], 4), semillas=3))
    ax[1].set_xlim(right=xmax * 4)
    ax[1].set_xscale("symlog", linthresh=0.01); ax[1].set_xlabel("Caída del HSS al desordenar el grupo (escala simétrica log)")
    ax[1].set_title("b) Importancia por permutación (modelo con todas las variables)")
else:
    ax[1].text(0.5, 0.5, f"No se encontró\n{imp}", ha="center", va="center", transform=ax[1].transAxes); ax[1].set_axis_off()
fig.suptitle("Precursores con la retención de 15 minutos (validación, α = 1.5, β = 10, mismos instantes)", y=1.02)
guardar(fig, "fig3_5_precursores_v6.png")

tv = pd.DataFrame(VALORES); tv.to_csv(SAL / "valores_figuras_v6.csv", index=False)
print(f"\nValores graficados: {SAL / 'valores_figuras_v6.csv'} ({len(tv)} filas)\nFIGURAS TERMINADAS")
