"""V08 · Re-análisis en PRUEBA del modelo ya elegido (v3.0) y de la variante B, SIN reentrenar.
Es un análisis posterior declarado: no cambia ningún modelo ni ninguna decisión.
Responde a:
  C1-016  métricas excluyendo además las ventanas con minutos reconstruidos en X;
  C1-022  inicios por EVENTO (no por ventana) y antelación de la primera alerta;
  C1-029  POFD = FP / (FP + VN) además de la razón de falsas alarmas;
  C1-030  intervalos de confianza (bootstrap por días) del HSS de cada modelo;
  C1-028  referencias más fuertes: persistencia con umbral calibrado en validación,
          climatología probabilística y AR(p) multipaso (umbrales calibrados en validación);
  C1-027  evaluación del v3.0 y de B en el periodo nunca usado 22/07–30/09/2026.
Instantes: los comunes a L = 30, 60 y 360 (los de la Tabla 4.8), para reproducir sus cifras.
Uso: python validacion/V08_reanalisis_prueba_v30.py [--v30 RUTA] [--b RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--b", default=B_DEFECTO)
ap.add_argument("--ar_p", type=int, default=20); ap.add_argument("--n_boot", type=int, default=1000); a = ap.parse_args()
cabecera("V08", a.config, "· PRUEBA: re-análisis declarado, sin decisiones")
from s4.config import cargar_config, ruta
from s4.pipeline import preparar_datos
from s4.windows import preparar_particion, inicios_comunes, inicios_validos
from s4.train import predecir
from s4 import evaluate as ev, io, features as ft
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
UMB = np.round(np.arange(0.30, 0.801, 0.05), 2)

MOD = {}
for nom, c in (("v3.0", a.v30), ("variante B", a.b)):
    if (Path(c) / "config.json").exists():
        MOD[nom] = cargar_modelo_app(c)
        print(f"{nom}: LOOKBACK={MOD[nom][2]['LOOKBACK']} · umbral={MOD[nom][2].get('UMBRAL_ALERTA')} · {c}")
Ls = sorted({30, 60, 360} | {m[2]["LOOKBACK"] for m in MOD.values()})
com = {k: inicios_comunes(d[k].index, Ls, H) for k in ("val", "test")}
def part(nom, k):
    c = MOD[nom][2]
    return preparar_particion(f"{k}·{nom}", d[k], MOD[nom][1], c["FEATURES_COLS"], c["LOOKBACK"], H,
                              cfg["datos"]["target"], cfg["datos"]["sampling_min"], hc, com[k][c["LOOKBACK"]])
P = {(n, k): part(n, k) for n in MOD for k in ("val", "test")}
ref = P[(next(iter(MOD)), "test")]; refv = P[(next(iter(MOD)), "val")]
for (n, k), p in P.items():
    assert (p.t == (ref if k == "test" else refv).t).all(), "instantes distintos"
PRED = {(n, k): predecir(MOD[n][0], P[(n, k)], MOD[n][1]) for (n, k) in P}

# --- referencias -------------------------------------------------------------------------------
def persist(p): return np.repeat(p.ultimo_real[:, None], H, axis=1)
s4 = d["df"]["S4"].to_numpy(); idx = d["df"].index
pos_t = lambda p: idx.get_indexer(p.t)
tr = d["train"]; st_tr = inicios_validos(tr.index, 60, H)
mtr = (tr["S4"].to_numpy()[st_tr[:, None] + 60 + np.arange(H)[None, :]].max(axis=1) >= 0.6)
t_tr = tr.index[st_tr + 59]
key = lambda t: (t.month * 10000 + t.hour * 60 + t.minute)
tabla_p = pd.Series(mtr.astype(float), index=key(t_tr)).groupby(level=0).mean()
tabla_h = pd.Series(mtr.astype(float), index=t_tr.hour * 60 + t_tr.minute).groupby(level=0).mean()
def climprob(p):
    k = key(pd.DatetimeIndex(p.t)); v = tabla_p.reindex(k).to_numpy()
    v = np.where(np.isnan(v), tabla_h.reindex(pd.DatetimeIndex(p.t).hour * 60 + pd.DatetimeIndex(p.t).minute).to_numpy(), v)
    return np.repeat(v[:, None], H, axis=1)
pp = a.ar_p; s_tr = tr["S4"].to_numpy(); st_ar = inicios_validos(tr.index, pp, 1)
Xa = np.c_[np.ones(len(st_ar)), s_tr[st_ar[:, None] + np.arange(pp)[None, :]]]; ya = s_tr[st_ar + pp]
coef = np.linalg.lstsq(Xa, ya, rcond=None)[0]
def ar(p):
    h = s4[pos_t(p)[:, None] - pp + 1 + np.arange(pp)[None, :]].copy(); out = []
    for _ in range(H):
        nxt = coef[0] + h @ coef[1:]; out.append(nxt); h = np.c_[h[:, 1:], nxt]
    return np.clip(np.stack(out, 1), 0, None)

def calibrar(p, yh):
    m = ~p.objetivo_reconstruido
    hss = np.array([ev.metricas_alerta(p.y_real[m], yh[m], 0.6, u)["HSS"] for u in UMB_REF], float)
    if np.isnan(hss).all():
        print("  ⚠ sin eventos en validación: no se puede calibrar; se usa 0.6"); return 0.6
    return float(UMB_REF[int(np.nanargmax(hss))])
REFS = {"Persistencia (0.6)": (persist, None), "Persistencia (umbral calibrado)": (persist, "cal"),
        "Climatología probabilística": (climprob, "cal_p"), f"AR({pp}) multipaso": (ar, "cal")}
for nom, (f, modo) in REFS.items():
    UMB_REF = np.round(np.arange(0.02, 0.6, 0.01), 2) if modo == "cal_p" else UMB
    yv, yt = f(refv), f(ref)
    u = 0.6 if modo is None else calibrar(refv, yv)
    PRED[(nom, "test")], PRED[(nom, "val")] = yt, yv; MOD[nom] = (None, None, {"UMBRAL_ALERTA": u})
    print(f"{nom}: umbral {'fijo' if modo is None else 'calibrado en validación'} = {u}")

# --- métricas ----------------------------------------------------------------------------------
pref = np.r_[0, np.cumsum(d["test"]["reconstruido"].to_numpy(bool))]
def sin_rec_x(p):                      # True si ningún minuto de la entrada X fue reconstruido
    return (pref[p.starts + p.lookback] - pref[p.starts]) == 0
def metr(p, yh, u, m):
    y = p.y_real[m]; yhm = yh[m]
    a_ = ev.metricas_alerta(y, yhm, 0.6, u); r_ = ev.metricas_regresion(y, yhm)
    evn = y.max(axis=1) >= .6; al = yhm.max(axis=1) >= u
    fp, tn = int((al & ~evn).sum()), int((~al & ~evn).sum())
    fase = ev.fase_ventana(p.ultimo_real[m], y); ini = fase == "inicio"
    return dict(n=int(m.sum()), HSS=a_["HSS"], POD=a_["POD"], Precision=a_["Precision"], FAR_razon=a_["FAR"],
                POFD=fp / (fp + tn) if fp + tn else np.nan, inicios_por_ventana=float(al[ini].mean()) if ini.any() else np.nan,
                RMSE_evento=r_["RMSE_evento"], RMSE_global=r_["RMSE_global"])
base_m = ~ref.objetivo_reconstruido
filas = []
for n in MOD:
    p = P.get((n, "test"), ref); yh = PRED[(n, "test")]; u = MOD[n][2]["UMBRAL_ALERTA"]
    filas.append(dict(modelo=n, umbral=u, mascara="tesis (sin objetivos reconstruidos)", **metr(p, yh, u, base_m)))
    filas.append(dict(modelo=n, umbral=u, mascara="además sin reconstruidos en X", **metr(p, yh, u, base_m & sin_rec_x(p))))
T1 = pd.DataFrame(filas)

# --- bootstrap por días del HSS de cada modelo ---------------------------------------------------
rng = np.random.default_rng(42); dias = pd.Series(ref.dia); grupos = [np.flatnonzero(dias.values == g) for g in dias.unique()]
fb = []
for n in MOD:
    yh = PRED[(n, "test")]; u = MOD[n][2]["UMBRAL_ALERTA"]; evn = ref.y_real.max(axis=1) >= .6; al = yh.max(axis=1) >= u
    vals = []
    for _ in range(a.n_boot):
        ii = np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))]); ii = ii[base_m[ii]]
        c = ev.contingencia(evn[ii], al[ii]); vals.append(c.get("HSS", np.nan) if isinstance(c, dict) else np.nan)
    fb.append(dict(modelo=n, HSS=ev.metricas_alerta(ref.y_real[base_m], yh[base_m], .6, u)["HSS"],
                   IC95_inf=float(np.nanpercentile(vals, 2.5)), IC95_sup=float(np.nanpercentile(vals, 97.5)), dias=len(grupos)))
T2 = pd.DataFrame(fb)

# --- inicios por EVENTO (C1-022) -----------------------------------------------------------------
fe = []
s_te = d["test"]["S4"]
for k in (0, 10, 15, 30):
    evs = eventos_unidos(s_te, 0.6, k)
    prev = s_te.reindex(evs["inicio"] - pd.Timedelta("1min"))
    inis = evs.loc[(prev.to_numpy() < 0.6), "inicio"]                    # inicio precedido por un minuto con dato < 0.6
    t = pd.DatetimeIndex(ref.t)
    for n in MOD:
        al = PRED[(n, "test")].max(axis=1) >= MOD[n][2]["UMBRAL_ALERTA"]
        cubiertos = alertados = 0; antel = []
        for tau in inis:
            w = (t >= tau - pd.Timedelta(minutes=H)) & (t < tau) & base_m
            if not w.any():
                continue
            cubiertos += 1
            if (al & w).any():
                alertados += 1; antel.append(int((tau - t[al & w].min()) / pd.Timedelta("1min")))
        fe.append(dict(k_union_min=k, modelo=n, inicios_de_evento=len(inis), inicios_con_ventanas=cubiertos,
                       inicios_con_alguna_alerta=alertados, pct_inicios_alertados=round(100 * alertados / cubiertos, 1) if cubiertos else None,
                       antelacion_primera_alerta_mediana_min=float(np.median(antel)) if antel else None,
                       antelacion_primera_alerta_p25=float(np.percentile(antel, 25)) if antel else None,
                       antelacion_primera_alerta_p75=float(np.percentile(antel, 75)) if antel else None))
T3 = pd.DataFrame(fe)

# --- periodo nunca usado 22/07–30/09/2026 (C1-027) ----------------------------------------------
serie = io.cargar_s4_reconstruido(ruta(cfg, "s4_reconstruido"), cfg["datos"]["flags_reconstruccion"])
fp_ = []
for n in [m for m in MOD if MOD[m][0] is not None]:
    modelo, esc, conf = MOD[n]; L = conf["LOOKBACK"]; u = conf["UMBRAL_ALERTA"]
    dfp = ft.agregar_ciclicas(serie.loc[pd.Timestamp("2026-07-22") - pd.Timedelta(minutes=L):"2026-09-30 23:59"])
    pp_ = preparar_particion(f"prospectivo·{n}", dfp, esc, conf["FEATURES_COLS"], L, H, "S4", 1, hc)
    ok = (pd.DatetimeIndex(pp_.t) >= "2026-07-22") & ~pp_.objetivo_reconstruido
    yh = predecir(modelo, pp_, esc)[ok]; y = pp_.y_real[ok]; t = pd.DatetimeIndex(pp_.t)[ok]
    al = yh.max(axis=1) >= u; evn = y.max(axis=1) >= .6
    fp_.append(dict(modelo=n, dias=int(t.normalize().nunique()), ventanas=int(ok.sum()), ventanas_con_evento=int(evn.sum()),
                    aciertos=int((al & evn).sum()), falsas_alarmas=int((al & ~evn).sum()), no_alertados=int((~al & evn).sum()),
                    episodios_de_alerta=int((al & ~np.r_[False, al[:-1]]).sum()), dias_con_alerta=int(t[al].normalize().nunique())))
T4 = pd.DataFrame(fp_)
out = [guardar(T1.round(4), "V08_metricas_prueba.csv", "Prueba: métricas (incluye referencias nuevas y máscara sin reconstruidos en X)"),
       guardar(T2.round(4), "V08_hss_ic95.csv", "HSS con IC 95 % (bootstrap por días)"),
       guardar(T3, "V08_inicios_por_evento.csv", "Inicios por EVENTO (k = minutos de unión: 0 = rachas de la tesis; 10, 15, 30 = rachas unidas)"),
       guardar(T4, "V08_prospectivo_22jul_30sep.csv", "Periodo nunca usado (22/07–30/09/2026)")]
bitacora("V08", a.config, out, "re-análisis declarado en prueba; ninguna decisión se toma con estas cifras")