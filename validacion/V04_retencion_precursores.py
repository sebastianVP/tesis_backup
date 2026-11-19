"""V04 · (C1-019) Diagnóstico de la retención de TEC, ROTEC y ROTI (antes y después de la corrección v6.0).

Reporta, por partición (entrenamiento, validación, prueba), sobre los minutos de S4 del periodo:
  1. Minutos de cada precursor rellenados MÁS ALLÁ de 15 min con la versión anterior
     (con dato en la versión anterior y sin dato con la retención de 15 min).
  2. Minutos de S4 sin precursor (algún precursor faltante) con cada versión: el pipeline
     los ELIMINA (dropna en construir_dataset), por lo que no entran en ninguna ventana.
  3. Ventanas válidas para L = 60 y L = 180 con solo S4 + hora (sin precursores) y con
     precursores (versión anterior y corregida), contadas por instante objetivo t.
No entrena modelos ni usa métricas; la partición de prueba solo se cuenta.

Uso:  python validacion/V04_retencion_precursores.py --config config/tesis_final.yaml (no requiere el parche)
"""
import argparse, subprocess, sys, datetime as dt
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from s4.config import cargar_config, ruta
from s4 import io, features as ft, splits
from s4.windows import inicios_validos
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/tesis_final.yaml"); a = ap.parse_args()
cfg = cargar_config(a.config); d = cfg["datos"]; H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "sin git"
except Exception:
    commit = "sin git"
print(f"Diagnóstico de retención v6.0 · {dt.datetime.now():%Y-%m-%d %H:%M} · commit {commit} · config {a.config}")

s4 = io.cargar_s4_reconstruido(ruta(cfg, "s4_reconstruido"), d["flags_reconstruccion"])
mv = io.cargar_multivariable(ruta(cfg, "multivariable"))
if cfg["rutas"].get("glotec"):
    mv = mv.drop(columns=["TEC", "ROTEC", "ROTI"]).join(io.cargar_glotec(ruta(cfg, "glotec")), how="left")

def causal(mv, recausalizar):
    m = mv.copy()
    for col, periodo in ft.PERIODO_INDICES.items():
        if col in m.columns:
            m[col] = ft.causalizar_indice(m[col], periodo)
    if recausalizar:
        for col in ft.LOCALES:
            if col in m.columns:
                m[col] = ft.causalizar_tramos_lineales(m[col], ft.diagnosticar_interpolacion(m[col])["nodos"])
    return m

viejo, nuevo = causal(mv, True), causal(mv, False)
base = io.recortar_periodo(s4[["S4"]], cfg["periodo"]["inicio"], cfg["periodo"]["fin"])
V = base.join(viejo, how="left"); N = base.join(nuevo, how="left")
cols = list(mv.columns); sp = cfg["split"]

def por_particion(df):
    tr, va, te = splits.particionar(df, sp["fin_train"], sp["fin_validacion"], hc)
    return {"entrenamiento": tr, "validación": va, "prueba": te}

PV, PN = por_particion(V), por_particion(N)
f1, f2 = [], []
for k in PV:
    v, n = PV[k], PN[k]
    fila = {"partición": k, "minutos_S4": len(v)}
    for col in ft.LOCALES:
        if col in cols:
            fila[f"{col}_rellenado_>15min_antes"] = int((v[col].notna() & n[col].isna()).sum())
    f1.append(fila)
    f2.append({"partición": k, "minutos_S4": len(v),
               "sin_precursor_antes": int(v[cols].isna().any(axis=1).sum()),
               "sin_precursor_corregido": int(n[cols].isna().any(axis=1).sum())})
t1, t2 = pd.DataFrame(f1), pd.DataFrame(f2)
t2["pct_sin_precursor_corregido"] = (100 * t2["sin_precursor_corregido"] / t2["minutos_S4"]).round(2)

def ventanas(df, L):
    st = inicios_validos(df.index, L, H, d["sampling_min"])
    return len(st)
f3 = []
for k in PV:
    solo = PN[k][["S4"]]
    for L in (60, 180):
        f3.append({"partición": k, "L": L, "ventanas_S4_hora": ventanas(solo, L),
                   "ventanas_con_precursores_antes": ventanas(PV[k].dropna(subset=cols), L),
                   "ventanas_con_precursores_corregido": ventanas(PN[k].dropna(subset=cols), L)})
t3 = pd.DataFrame(f3)
t3["perdidas_corregido"] = t3["ventanas_S4_hora"] - t3["ventanas_con_precursores_corregido"]
t3["pct_perdidas_corregido"] = (100 * t3["perdidas_corregido"] / t3["ventanas_S4_hora"]).round(2)

pd.set_option("display.width", 220)
print("\n=== 1. Minutos rellenados más allá de 15 min con la versión anterior ===\n" + t1.to_string(index=False))
print("\n=== 2. Minutos de S4 sin precursor (se eliminan del conjunto con precursores) ===\n" + t2.to_string(index=False))
print("\n=== 3. Ventanas válidas por partición ===\n" + t3.to_string(index=False))
out = Path("validacion/salidas"); out.mkdir(parents=True, exist_ok=True)
t1.to_csv(out / "V6_relleno_mas_alla_15min.csv", index=False)
t2.to_csv(out / "V6_minutos_sin_precursor.csv", index=False)
t3.to_csv(out / "V6_ventanas_por_particion.csv", index=False)
print(f"\nTablas en {out}\nDIAGNÓSTICO TERMINADO")
