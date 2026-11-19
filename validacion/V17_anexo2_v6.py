"""V17 · Tablas A2.1 a A2.9 del Anexo 2 con la repetición v6 (retención de 15 min).
Mismas columnas que la tesis. Lee experiments/registro_tesis_v6.csv (validación).
Uso (desde la raíz del proyecto): python validacion/V17_anexo2_v6.py
Salida: validacion/salidas/V17_A2_<n>.csv y un resumen en pantalla para copiar."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--registro", default="experiments/registro_tesis_v6.csv")
ap.add_argument("--salida", default="validacion/salidas"); a = ap.parse_args()
R = pd.read_csv(a.registro); R = R[R["particion"] == "val"].copy()
SAL = Path(a.salida); SAL.mkdir(parents=True, exist_ok=True)
pct = lambda v: "—" if pd.isna(v) else f"{100 * v:.0f} %"
num = lambda v, d=3: "—" if pd.isna(v) else f"{v:.{d}f}"
ent = lambda v: "—" if pd.isna(v) else f"{int(v)}"
col = lambda x, c: x[c] if c in x else np.nan

def fija(g, nombre, titulo, primera="Modelo"):
    x = R[R["grupo"] == g]
    if x.empty: print(f"(sin filas para {g})"); return
    filas = [{primera: r["modelo"], "Semilla": ent(col(r, "semilla")), "HSS": num(col(r, "HSS")), "POD": pct(col(r, "POD")),
              "Precisión": pct(col(r, "Precision")), "Inicios": pct(col(r, "POD_inicio")), "RMSE eventos": num(col(r, "RMSE_evento")),
              "RMSE global": num(col(r, "RMSE_global"), 4), "Épocas": ent(col(r, "epocas"))} for _, r in x.iterrows()]
    salida(pd.DataFrame(filas), nombre, titulo)

def calibrada(g, nombre, titulo):
    x = R[R["grupo"] == g]
    if x.empty: print(f"(sin filas para {g})"); return
    filas = [{"Modelo": r["modelo"], "Semilla": ent(col(r, "semilla")), "Umbral": num(col(r, "umbral_opt"), 2), "HSS calibrado": num(col(r, "HSS_opt")),
              "Precisión": pct(col(r, "Precision_opt")), "Inicios": pct(col(r, "POD_inicio_opt")), "RMSE eventos": num(col(r, "RMSE_evento")),
              "RMSE global": num(col(r, "RMSE_global"), 4), "Épocas": ent(col(r, "epocas"))} for _, r in x.iterrows()]
    salida(pd.DataFrame(filas), nombre, titulo)

def salida(df, nombre, titulo):
    df.to_csv(SAL / f"V17_{nombre}.csv", index=False)
    print(f"\n=== {nombre} · {titulo} ({len(df)} filas)\n" + df.to_string(index=False))

fija("E2_ventana", "A2_1", "Ventana de historia (S4 + hora, β = 100, umbral 0.6)")
fija("E3_ablacion", "A2_2", "Ablación con ventana de 60 min")
fija("E3b_ablacion_extra", "A2_3", "Ablación complementaria")
x = R[R["grupo"] == "E4_fase1"].copy()
if not x.empty:
    x["α"] = x["modelo"].str.extract(r"a([\d.]+)_")[0]; x["β"] = x["modelo"].str.extract(r"_b(\d+)")[0].astype(int)
    x = x.sort_values(["α", "β"])
    salida(pd.DataFrame([{"α": r["α"], "β": r["β"], "HSS": num(col(r, "HSS")), "POD": pct(col(r, "POD")), "Precisión": pct(col(r, "Precision")),
                          "Inicios": pct(col(r, "POD_inicio")), "RMSE eventos": num(col(r, "RMSE_evento")), "RMSE global": num(col(r, "RMSE_global"), 4),
                          "Épocas": ent(col(r, "epocas"))} for _, r in x.iterrows()]), "A2_4", "Fase 1: grilla β × α (semilla 42, umbral 0.6)")
calibrada("E4_fase2", "A2_5", "Fase 2: β con umbral calibrado (α = 1.5)")
calibrada("E4_fase3", "A2_6", "Fase 3: α con β = 10 y umbral calibrado")
calibrada("E5_patchtst", "A2_7", "LSTM apilada frente a PatchTST")
calibrada("E6_precursores", "A2_8", "Precursores con datos corregidos")
calibrada("E6_explicabilidad", "A2_9", "Modelo del análisis de explicabilidad (D', 180 min)")
print(f"\nCSV en {SAL}/V17_A2_*.csv · registro: {a.registro}")
