"""V07 · Autocorrelación del S4: total, nocturna y sin ciclo diario (C1-031).
Calcula la ACF hasta 120 min en el periodo experimental (2025-01-26 a 2026-07-21):
(a) serie completa; (b) solo horas nocturnas de Lima (23:00–11:00 UT);
(c) anomalía = S4 − climatología por minuto del día (ajustada en entrenamiento).
Reporta el primer retardo con ACF < 0.5 en cada caso.
Uso: python validacion/V07_autocorrelacion.py"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); a = ap.parse_args()
cabecera("V07", a.config)
from s4.config import cargar_config
from s4.pipeline import preparar_datos
cfg = cargar_config(a.config); d = preparar_datos(cfg, usar_precursores=False, verbose=False)
s = d["df"]["S4"].asfreq("1min")
clim = d["train"]["S4"].groupby(d["train"].index.hour * 60 + d["train"].index.minute).mean()
anom = s - clim.reindex(s.index.hour * 60 + s.index.minute).to_numpy()
noct = s.where((s.index.hour >= 23) | (s.index.hour < 11))
def acf(x, L=120):
    return [x.corr(x.shift(k)) for k in range(1, L + 1)]
t = pd.DataFrame({"lag_min": range(1, 121), "ACF_total": acf(s), "ACF_nocturna": acf(noct), "ACF_anomalia": acf(anom)})
prim = {c: int(t.loc[t[c] < 0.5, "lag_min"].min()) if (t[c] < 0.5).any() else ">120" for c in t.columns[1:]}
out = [guardar(t.round(4), "V07_acf.csv"), guardar(pd.DataFrame([prim]), "V07_primer_lag_acf_lt05.csv", "Primer retardo con ACF < 0.5")]
bitacora("V07", a.config, out)
