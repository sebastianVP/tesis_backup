"""V13b · Tablas de la tesis (3.2, 3.3, 3.4, 3.6, 3.7) con los resultados v6 (retención de 15 min), en el formato del texto.
Uso: python validacion/V13b_tablas_tesis_v6.py"""
from comun import *
r = pd.read_csv("experiments/registro_tesis_v6.csv"); r = r[r.particion == "val"]
def ms(x, c, d=3, pct=False):
    if c not in x or x[c].isna().all(): return "—"
    m, s = x[c].mean(), x[c].std()
    return f"{100*m:.0f} %" if pct else (f"{m:.{d}f}" + (f" ± {s:.{d}f}" if len(x) > 1 and not np.isnan(s) else ""))
def tabla(g, cols, nombre, titulo):
    filas = []
    for mod, x in r[r.grupo == g].groupby("modelo", sort=False):
        filas.append({"modelo": mod, **{k: ms(x, c, **kw) for k, (c, kw) in cols.items()}})
    return guardar(pd.DataFrame(filas), nombre, titulo)
P = lambda c: (c, {"pct": True}); N = lambda c, d=3: (c, {"d": d})
out = [tabla("E2_ventana", {"HSS": N("HSS"), "RMSE_ev": N("RMSE_evento"), "POD": P("POD"), "Prec": P("Precision"), "Inicios": P("POD_inicio")}, "V13b_tabla_3_2.csv", "Tabla 3.2 (E2)"),
       tabla("E3_ablacion", {"HSS": N("HSS"), "POD": P("POD"), "Prec": P("Precision"), "Inicios": P("POD_inicio"), "HSS_cal": N("HSS_opt"), "umbral": N("umbral_opt", 2)}, "V13b_tabla_3_3.csv", "Tabla 3.3 (E3)"),
       tabla("E3b_ablacion_extra", {"HSS": N("HSS"), "HSS_cal": N("HSS_opt")}, "V13b_tabla_3_3_extra.csv", "E3b"),
       tabla("E4_fase2", {"HSS_cal": N("HSS_opt"), "umbral": N("umbral_opt", 2), "Prec": P("Precision_opt"), "Inicios": P("POD_inicio_opt"), "RMSE_ev": N("RMSE_evento"), "RMSE_gl": N("RMSE_global")}, "V13b_tabla_3_4.csv", "Tabla 3.4 (E4 fase 2)"),
       tabla("E4_fase3", {"HSS_cal": N("HSS_opt")}, "V13b_E4_fase3.csv", "E4 fase 3"),
       tabla("E5_patchtst", {"HSS_cal": N("HSS_opt"), "Prec": P("Precision_opt"), "RMSE_ev": N("RMSE_evento"), "params": N("parametros", 0)}, "V13b_tabla_3_6.csv", "Tabla 3.6 (E5)"),
       tabla("E6_precursores", {"HSS_cal": N("HSS_opt"), "Prec": P("Precision_opt"), "Inicios": P("POD_inicio_opt")}, "V13b_tabla_3_7.csv", "Tabla 3.7 (E6)")]
x = r[r.grupo == "E5_patchtst"]
print("\nE5 por semilla:\n" + x[["modelo", "semilla", "HSS_opt"]].round(3).to_string(index=False))
for f in ("reports/tables/V6_E6_saliencia_inicio.csv", "reports/tables/V6_E5_latencia.csv"):
    if Path(f).exists():
        print(f"\n--- {f} ---\n" + pd.read_csv(f).round(4).to_string(index=False))
bitacora("V13b", "-", out)
