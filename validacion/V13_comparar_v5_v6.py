"""V13 · Comparación de los experimentos con precursores antes (v5, retención sin límite)
y después (v6, retención de 15 min), en VALIDACIÓN, y respuestas a las preguntas a)–d).
Lee experiments/registro_tesis_final.csv (antes) y experiments/registro_tesis_v6.csv (después).
Uso: python validacion/V13_comparar_v5_v6.py"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--antes", default="experiments/registro_tesis_final.csv")
ap.add_argument("--despues", default="experiments/registro_tesis_v6.csv"); a = ap.parse_args()
cabecera("V13", "-")
A, D = pd.read_csv(a.antes), pd.read_csv(a.despues)
A, D = A[A.particion == "val"], D[D.particion == "val"]
MET = ["HSS", "POD", "Precision", "POD_inicio", "HSS_opt", "Precision_opt", "POD_inicio_opt", "umbral_opt"]
def res(df, g):
    x = df[df.grupo == g]
    if x.empty:
        return pd.DataFrame()
    cols = [c for c in MET if c in x.columns and x[c].notna().any()]
    r = x.groupby("modelo")[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=r.index)
    for c in cols:
        out[c] = r[(c, "mean")].round(3).astype(str) + " ± " + r[(c, "std")].round(3).fillna(0).astype(str)
    return out
out = []
for g in sorted(set(D.grupo)):
    ra, rd = res(A, g), res(D, g)
    t = ra.add_suffix(" (antes)").join(rd.add_suffix(" (v6)"), how="outer").reset_index()
    out.append(guardar(t, f"V13_{g}_antes_v6.csv", f"{g}: antes frente a v6 (validación)"))

def regla(df, g, ref_nombre, met="HSS_opt"):
    x = df[df.grupo == g]; r = x[x.modelo == ref_nombre].set_index("semilla")[met]; filas = []
    for m, gg in x.groupby("modelo"):
        if m == ref_nombre: continue
        v = gg.set_index("semilla")[met]; c = v.index.intersection(r.index)
        filas.append(dict(grupo=g, modelo=m, referencia=ref_nombre, dif_media=round(v.mean() - r.mean(), 3),
                          tres_semillas_mejoran=bool(len(c) == 3 and (v[c] > r[c]).all()),
                          supera_regla=bool(v.mean() - r.mean() > 0.03 and len(c) == 3 and (v[c] > r[c]).all())))
    return filas
resp = []
for L in (60, 180):
    resp += [f for f in regla(D, "E6_precursores", f"C · S4 + hora · LB{L}") if f"LB{L}" in f["modelo"]]
resp += regla(D, "E3_ablacion", "C · + hora + WFL", "HSS_opt" if "HSS_opt" in D.columns else "HSS")
resp += regla(D, "E3b_ablacion_extra", "C · + hora · LB180", "HSS_opt" if "HSS_opt" in D.columns else "HSS")
out.append(guardar(pd.DataFrame(resp), "V13_a_precursores_vs_C.csv", "a) ¿Algún precursor supera a C (> 0.03 y 3 semillas)?"))
imp = Path("reports/tables/V6_E6_importancia_Dprima.csv")
if imp.exists():
    out.append(guardar(pd.read_csv(imp).round(4), "V13_b_importancia_ROTI.csv", "b) Importancia por permutación en D' (v6)"))
f2 = res(D, "E4_fase2"); f3 = res(D, "E4_fase3")
if not f2.empty:
    x = D[D.grupo == "E4_fase2"].groupby("modelo")["HSS_opt"].agg(["mean", "std"])
    mejor = x["mean"].idxmax(); meseta = x[x["mean"] >= x.loc[mejor, "mean"] - x.loc[mejor, "std"]].index.tolist()
    out.append(guardar(pd.DataFrame([dict(mejor_beta=mejor, betas_a_1_desv_del_mejor=", ".join(meseta))]), "V13_c_meseta_beta.csv",
                       "c) Meseta de β (dentro de 1 desviación del mejor HSS calibrado)"))
x = D[D.grupo == "E5_patchtst"]
if not x.empty:
    out.append(guardar(x.groupby("modelo")["HSS_opt"].agg(["mean", "std"]).round(3).reset_index(), "V13_d_lstm_patchtst.csv",
                       "d) LSTM apilada frente a PatchTST (HSS calibrado)"))
bitacora("V13", "-", out)
