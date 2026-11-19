"""V21 · Dos verificaciones pedidas por la auditoría de la v6.2.

(A) C3-002 · Anticipación por episodio SIN contaminación de episodios previos (prueba, re-análisis declarado).
    Con la unión de rachas ≤ 15 min, un episodio puede empezar 16–20 min después del anterior, y una alerta emitida
    durante el final del anterior cae en los 20 min previos al nuevo inicio. Aquí solo se cuentan los episodios
    «limpios»: sin ningún minuto con S4 ≥ 0.6 en los 20 minutos previos a su inicio. Se informa cuántos se descartan.
    Control: la persistencia con umbral 0.6 debe dar 0 episodios anticipados.
    Modelos: v3.0 (0.55), variante B (0.50), persistencia (0.6), regla de S4 actual (0.40) y AR(20) (0.40),
    con los umbrales calibrados en validación ya reportados (V08). Mismos instantes que la Tabla 3.8.

(B) C1-034 · Huecos de la serie de S4: longitud de los tramos reconstruidos (Fix1, Fix3, Fix4) y de los huecos que
    quedaron sin reconstruir dentro de cada día UT, por clases (≤ 5, 6–60, 61–180, > 180 minutos).

Uso: python validacion/V21_episodios_limpios_y_huecos.py [--v30 RUTA] [--b RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--b", default=B_DEFECTO); a = ap.parse_args()
cabecera("V21", a.config, "· (A) PRUEBA: re-análisis declarado · (B) solo datos")
from s4.config import cargar_config, ruta
from s4.pipeline import preparar_datos
from s4.windows import preparar_particion, inicios_comunes, inicios_validos
from s4.train import predecir
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
d = preparar_datos(cfg, usar_precursores=False, verbose=False)

# ---------------- (A) episodios limpios ----------------
MOD = {n: cargar_modelo_app(c) for n, c in (("v3.0", a.v30), ("variante B", a.b))}
Ls = sorted({30, 60, 360} | {m[2]["LOOKBACK"] for m in MOD.values()})
com = inicios_comunes(d["test"].index, Ls, H)
P = {n: preparar_particion(f"test·{n}", d["test"], m[1], m[2]["FEATURES_COLS"], m[2]["LOOKBACK"], H, "S4",
                           cfg["datos"]["sampling_min"], hc, com[m[2]["LOOKBACK"]]) for n, m in MOD.items()}
ref = P["v3.0"]; base_m = ~ref.objetivo_reconstruido; t = pd.DatetimeIndex(ref.t)
PRED = {n: predecir(MOD[n][0], P[n], MOD[n][1]) for n in MOD}
U = {"v3.0": MOD["v3.0"][2]["UMBRAL_ALERTA"], "variante B": MOD["variante B"][2]["UMBRAL_ALERTA"]}
PRED["Persistencia (0.6)"] = np.repeat(ref.ultimo_real[:, None], H, axis=1); U["Persistencia (0.6)"] = 0.6
PRED["Regla de S4 actual (0.40)"] = PRED["Persistencia (0.6)"]; U["Regla de S4 actual (0.40)"] = 0.40
s4 = d["df"]["S4"].to_numpy(); idx = d["df"].index; tr = d["train"]["S4"].to_numpy(); pp = 20
st = inicios_validos(d["train"].index, pp, 1)
coef = np.linalg.lstsq(np.c_[np.ones(len(st)), tr[st[:, None] + np.arange(pp)]], tr[st + pp], rcond=None)[0]
h = s4[idx.get_indexer(ref.t)[:, None] - pp + 1 + np.arange(pp)[None, :]].copy(); out = []
for _ in range(H):
    nx = coef[0] + h @ coef[1:]; out.append(nx); h = np.c_[h[:, 1:], nx]
PRED["AR(20) (0.40)"] = np.clip(np.stack(out, 1), 0, None); U["AR(20) (0.40)"] = 0.40

s_te = d["test"]["S4"]; filas = []
for k in (15, 30):
    evs = eventos_unidos(s_te, 0.6, k)
    prev = s_te.reindex(evs["inicio"] - pd.Timedelta("1min"))
    inis = evs.loc[prev.to_numpy() < 0.6, "inicio"]
    limpio = [not (s_te[(s_te.index >= tau - pd.Timedelta(minutes=H)) & (s_te.index < tau)] >= 0.6).any() for tau in inis]
    inis_l = inis[np.array(limpio, bool)]
    for n in PRED:
        al = PRED[n].max(axis=1) >= U[n]; cub = alert = 0; antel = []
        for tau in inis_l:
            w = (t >= tau - pd.Timedelta(minutes=H)) & (t < tau) & base_m
            if not w.any(): continue
            cub += 1
            if (al & w).any():
                alert += 1; antel.append(int((tau - t[al & w].min()) / pd.Timedelta("1min")))
        filas.append(dict(k_union_min=k, modelo=n, episodios=len(inis), descartados_por_episodio_previo=len(inis) - len(inis_l),
                          episodios_limpios_con_ventanas=cub, alertados=alert, pct_alertados=round(100 * alert / cub, 1) if cub else None,
                          antelacion_mediana_min=float(np.median(antel)) if antel else None,
                          antelacion_p25=float(np.percentile(antel, 25)) if antel else None, antelacion_p75=float(np.percentile(antel, 75)) if antel else None))
A = pd.DataFrame(filas)
out = [guardar(A, "V21_episodios_limpios.csv", "(A) Anticipación por episodio sin contaminación de episodios previos (prueba)")]
print("Control (A): «Persistencia (0.6)» debe tener 0 episodios alertados.")

# ---------------- (B) huecos ----------------
from s4 import io as s4io
raw = s4io._a_indice_temporal(pd.read_csv(ruta(cfg, "s4_reconstruido"), index_col=None)).sort_index()   # misma lectura que el paquete
rec = np.zeros(len(raw), bool)
for f in ("Fix1", "Fix3", "Fix4"):
    if f in raw.columns: rec |= raw[f].astype(str).str.lower().isin(["true", "1"]).to_numpy()
clases = [(1, 5, "≤ 5"), (6, 60, "6–60"), (61, 180, "61–180"), (181, 10**9, "> 180")]
def runs(mask, tt):
    r = []; n = 0
    for i, v in enumerate(mask):
        if v and n and (tt[i] - tt[i - 1]) == pd.Timedelta("1min"): n += 1
        elif v: (r.append(n) if n else None); n = 1
        else: (r.append(n) if n else None); n = 0
    if n: r.append(n)
    return np.array(r)
tr_rec = runs(rec, raw.index)
huecos = []
for dia, g in raw.groupby(raw.index.normalize()):
    full = pd.date_range(dia, dia + pd.Timedelta("1439min"), freq="1min")
    falta = ~full.isin(g.index)
    if falta.all(): continue                                   # día sin ningún dato (no es un hueco intradiario)
    huecos.extend(runs(falta, full).tolist())
huecos = np.array(huecos)
B = pd.DataFrame([dict(clase=c, tramos_reconstruidos=int(((tr_rec >= lo) & (tr_rec <= hi)).sum()),
                       minutos_reconstruidos=int(tr_rec[(tr_rec >= lo) & (tr_rec <= hi)].sum()),
                       huecos_sin_reconstruir=int(((huecos >= lo) & (huecos <= hi)).sum()),
                       minutos_sin_reconstruir=int(huecos[(huecos >= lo) & (huecos <= hi)].sum())) for lo, hi, c in clases])
out.append(guardar(B, "V21_huecos_por_clase.csv", "(B) Tramos reconstruidos y huecos intradiarios sin reconstruir, por longitud"))
print(f"\nTramo reconstruido más largo: {tr_rec.max() if len(tr_rec) else 0} min · tesis: 40 huecos > 180 min (16 340 min), 15 814 minutos reconstruidos (incluye 8 de Fix0)")
bitacora("V21", a.config, out)