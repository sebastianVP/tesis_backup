"""V08b · ¿El v3.0 supera a las referencias calibradas? (continuación de C1-028 / C1-030)
Compara, en VALIDACIÓN y en PRUEBA, el v3.0 y la variante B con la persistencia y el AR(p)
multipaso, ambos con umbral calibrado en validación (el mismo trato que los modelos).
En prueba, diferencias pareadas con bootstrap por días (HSS alineado y RMSE en eventos).
Re-análisis declarado: no cambia el modelo ni ninguna decisión.
Uso: python validacion/V08b_significancia_referencias.py [--v30 RUTA] [--b RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--b", default=B_DEFECTO)
ap.add_argument("--ar_p", type=int, default=20); ap.add_argument("--n_boot", type=int, default=1000); a = ap.parse_args()
cabecera("V08b", a.config, "· PRUEBA: re-análisis declarado")
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import preparar_particion, inicios_comunes, inicios_validos
from s4.train import predecir
from s4 import evaluate as ev, stats
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
MOD = {"v3.0": cargar_modelo_app(a.v30), "variante B": cargar_modelo_app(a.b)}
com = {k: inicios_comunes(d[k].index, [30, 60, 360], H) for k in ("val", "test")}
P, PRED, U = {}, {}, {}
for n, (m, esc, c) in MOD.items():
    for k in ("val", "test"):
        P[(n, k)] = preparar_particion(f"{k}·{n}", d[k], esc, c["FEATURES_COLS"], c["LOOKBACK"], H, "S4", 1, hc, com[k][c["LOOKBACK"]])
        PRED[(n, k)] = predecir(m, P[(n, k)], esc)
    U[n] = c["UMBRAL_ALERTA"]
ref = {k: P[("v3.0", k)] for k in ("val", "test")}
s4 = d["df"]["S4"].to_numpy(); idx = d["df"].index; pp = a.ar_p
tr = d["train"]["S4"].to_numpy(); st = inicios_validos(d["train"].index, pp, 1)
coef = np.linalg.lstsq(np.c_[np.ones(len(st)), tr[st[:, None] + np.arange(pp)[None, :]]], tr[st + pp], rcond=None)[0]
def ar(p):
    h = s4[idx.get_indexer(p.t)[:, None] - pp + 1 + np.arange(pp)[None, :]].copy(); out = []
    for _ in range(H):
        nx = coef[0] + h @ coef[1:]; out.append(nx); h = np.c_[h[:, 1:], nx]
    return np.clip(np.stack(out, 1), 0, None)
persist = lambda p: np.repeat(p.ultimo_real[:, None], H, axis=1)
UMB = np.round(np.arange(0.30, 0.801, 0.05), 2)
def calibrar(p, yh):
    m = ~p.objetivo_reconstruido
    h = np.array([ev.metricas_alerta(p.y_real[m], yh[m], .6, u)["HSS"] for u in UMB], float)
    return float(UMB[int(np.nanargmax(h))])
for n, f in (("Persistencia calibrada", persist), (f"AR({pp}) calibrado", ar)):
    for k in ("val", "test"):
        PRED[(n, k)] = f(ref[k])
    U[n] = calibrar(ref["val"], PRED[(n, "val")])
filas = []
for k in ("val", "test"):
    m = ~ref[k].objetivo_reconstruido
    for n in U:
        a_ = ev.metricas_alerta(ref[k].y_real[m], PRED[(n, k)][m], .6, U[n]); r_ = ev.metricas_regresion(ref[k].y_real[m], PRED[(n, k)][m])
        filas.append(dict(particion=k, modelo=n, umbral=U[n], HSS=a_["HSS"], POD=a_["POD"], Precision=a_["Precision"],
                          RMSE_evento=r_["RMSE_evento"], RMSE_global=r_["RMSE_global"]))
T1 = pd.DataFrame(filas)
p = ref["test"]; m = ~p.objetivo_reconstruido; al = lambda n: PRED[(n, "test")] - U[n] + 0.6
sig = []
for mod in ("v3.0", "variante B"):
    for r in ("Persistencia calibrada", f"AR({pp}) calibrado"):
        bh = stats.bootstrap_bloques(p.y_real, al(r), al(mod), p.dia, stats.m_menos_hss(.6, .6), n_boot=a.n_boot, mascara=m)
        br = stats.bootstrap_bloques(p.y_real, PRED[(r, "test")], PRED[(mod, "test")], p.dia, stats.m_rmse_evento(), n_boot=a.n_boot, mascara=m)
        sig.append(dict(modelo=mod, referencia=r, mejora_HSS=round(bh["diferencia"], 4), IC95_HSS=tuple(round(x, 4) for x in bh["ic95"]),
                        sig_HSS=bh["significativo"], mejora_RMSE_evento=round(br["diferencia"], 4),
                        IC95_RMSE_evento=tuple(round(x, 4) for x in br["ic95"]), sig_RMSE=br["significativo"]))
out = [guardar(T1.round(4), "V08b_referencias_val_test.csv", "Validación y prueba (umbrales calibrados en validación)"),
       guardar(pd.DataFrame(sig), "V08b_significancia_vs_referencias.csv",
               "Prueba: diferencias pareadas (positivo = el modelo es mejor; bootstrap por días)")]
bitacora("V08b", a.config, out, "re-análisis declarado en prueba")