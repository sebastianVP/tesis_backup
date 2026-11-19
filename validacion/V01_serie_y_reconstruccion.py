"""V01 · Serie de S4 y reconstrucción (C1-017, C1-018, Tabla 1.4).
Reproduce desde df_max_s4_reconstruido.csv: rango, días con/sin dato, minutos por
etapa Fix0–Fix4, cobertura diaria y concentración horaria de lo reconstruido.
Con --a2 (df_max_s4_all_jic.csv, antes de reconstruir) cuenta además los días con
cobertura < 50 % en la serie completa y en el rango de E1 (01/01/2022–31/08/2025),
para conciliar los 71 días de la tesis con los 326 del análisis de E1.
Uso: python validacion/V01_serie_y_reconstruccion.py [--a2 RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--a2", default=str(Path.home() / "Documents/pipelines_cip/S4_LISN/processed_s4/df_max_s4_all_jic.csv"))
ap.add_argument("--cierre", default="2026-09-30 23:59"); a = ap.parse_args()
cabecera("V01", a.config)
from s4.config import cargar_config, ruta
cfg = cargar_config(a.config)
rec = pd.read_csv(ruta(cfg, "s4_reconstruido"), index_col=0, parse_dates=True).loc[:a.cierre]
flags = [f for f in ("Fix0", "Fix1", "Fix3", "Fix4", "Fix_Pico") if f in rec.columns]
for f in flags:
    rec[f] = rec[f].astype(str).str.lower().isin(["true", "1"])
dias_cal = pd.date_range(rec.index.min().normalize(), rec.index.max().normalize(), freq="D")
dias_dato = rec.index.normalize().unique()
cob = rec.groupby(rec.index.normalize()).size() / 1440
r = rec[flags[1:]].any(axis=1) if len(flags) > 1 else pd.Series(False, index=rec.index)
resumen = pd.DataFrame([dict(inicio=rec.index.min(), fin=rec.index.max(), minutos=len(rec),
                             dias_calendario=len(dias_cal), dias_con_dato=len(dias_dato),
                             dias_sin_ningun_dato=len(dias_cal) - len(dias_dato),
                             dias_cobertura_ge90=int((cob >= .9).sum()), pct_dias_ge90=round(100 * (cob >= .9).mean(), 2),
                             cobertura_minima_pct=round(100 * cob.min(), 1),
                             minutos_reconstruidos=int(r.sum()), pct_reconstruidos=round(100 * r.mean(), 3),
                             max_S4_reconstruido=float(rec.loc[r, "S4"].max()) if r.any() else None)])
etapas = pd.DataFrame([dict(etapa=f, minutos=int(rec[f].sum()), dias=int(rec.index[rec[f]].normalize().nunique()),
                            S4_min=float(rec.loc[rec[f], "S4"].min()) if rec[f].any() else None,
                            S4_max=float(rec.loc[rec[f], "S4"].max()) if rec[f].any() else None) for f in flags])
h = rec.index[r]
franja = ((h.hour == 23) & (h.minute >= 46)) | ((h.hour == 0) & (h.minute == 0))
horas = pd.DataFrame(dict(hora_UT=range(24), minutos_reconstruidos=[int((h.hour == k).sum()) for k in range(24)]))
conc = pd.DataFrame([dict(minutos_reconstruidos=int(r.sum()), en_23_46_a_00_00_UT=int(franja.sum()),
                          pct_en_franja=round(100 * franja.mean(), 2) if len(h) else None)])
out = [guardar(resumen, "V01_resumen_serie.csv", "Resumen de la serie reconstruida"),
       guardar(etapas, "V01_etapas_fix.csv", "Etapas de reconstrucción"),
       guardar(conc, "V01_reconstruidos_franja_2346.csv", "Concentración en 23:46–00:00 UT"),
       guardar(horas, "V01_reconstruidos_por_hora.csv")]
if Path(a.a2).exists():
    a2 = pd.read_csv(a.a2, usecols=lambda c: c in ("Datetime", "S4"), parse_dates=["Datetime"]).set_index("Datetime")
    c2 = a2.groupby(a2.index.normalize()).size() / 1440
    filas = []
    for nom, ini, fin in (("serie tesis (cierre 30/09/2026)", "2022-01-01", "2026-09-30"), ("rango E1 (2022-01-01 a 2025-08-31)", "2022-01-01", "2025-08-31")):
        cc = c2.loc[ini:fin]; cal = pd.date_range(ini, fin, freq="D")
        filas.append(dict(rango=nom, dias_calendario=len(cal), dias_con_dato=len(cc), dias_sin_dato=len(cal) - len(cc),
                          dias_cobertura_lt50_antes_de_reconstruir=int((cc < .5).sum())))
    out.append(guardar(pd.DataFrame(filas), "V01_conciliacion_dias_lt50.csv",
                       "Conciliación de días con cobertura < 50 % (A2, antes de reconstruir; la tesis filtra DESPUÉS de Fix1)"))
else:
    print(f"\n(No se encontró {a.a2}: se omite la conciliación 71 frente a 326)")
bitacora("V01", a.config, out)
