"""V06 · Elevación del satélite que produce el máximo y cambios de satélite (C1-014).
Necesita la tabla por satélite y minuto limpia de A2 (s4_base). Para cada minuto toma el
satélite con mayor S4 y, en los minutos con S4 máximo >= 0.6, reporta la distribución de
su elevación, la constelación y cuántas veces cambia el satélite dentro de cada evento.
Uso: python validacion/V06_elevacion_satelite_maximo.py --base RUTA_s4_base.csv [--desde 2025-01-26 --hasta 2026-07-21]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True)
ap.add_argument("--desde", default="2025-01-26"); ap.add_argument("--hasta", default="2026-07-21 23:59"); a = ap.parse_args()
cabecera("V06", "-", a.base)
cols = pd.read_csv(a.base, nrows=2).columns
pick = lambda *ops: next((c for c in cols if c.upper() in ops), None)
ct, cs, ce, cp = pick("TIME", "DATETIME", "TIEMPO"), pick("S4"), pick("ELEVATION", "ELEVACION", "ELEV"), pick("ID_SATELLITE", "ID_SATELITE", "PRN", "SAT")
cc = pick("CONSTELACION", "CONSTELLATION")
if None in (ct, cs, ce, cp):
    sys.exit(f"No se identificaron las columnas necesarias en {list(cols)}")
partes = []
for ch in pd.read_csv(a.base, usecols=[c for c in (ct, cs, ce, cp, cc) if c], parse_dates=[ct], chunksize=2_000_000):
    ch = ch[(ch[ct] >= a.desde) & (ch[ct] <= a.hasta)]
    if len(ch):
        partes.append(ch.loc[ch.groupby(ct)[cs].idxmax()])
mx = pd.concat(partes).sort_values(ct)
mx = mx.loc[mx.groupby(ct)[cs].idxmax()].set_index(ct)
ev = mx[mx[cs] >= 0.6]
bins = [30, 40, 50, 60, 70, 80, 90]
t1 = (pd.cut(ev[ce], bins, right=False).value_counts().sort_index().rename_axis("elevacion").reset_index(name="minutos"))
t1["pct"] = (100 * t1["minutos"] / t1["minutos"].sum()).round(1)
t2 = pd.DataFrame([dict(minutos_S4max_ge06=len(ev), elevacion_mediana=float(ev[ce].median()),
                        pct_elev_lt40=round(100 * (ev[ce] < 40).mean(), 1),
                        **({"constelaciones": ev[cc].value_counts().to_dict()} if cc else {}))])
evs = eventos_unidos(mx[cs], 0.6, 0); cambios = []
for _, e in evs.iterrows():
    prn = mx.loc[e["inicio"]:e["fin"], cp]; prn = prn[mx.loc[e["inicio"]:e["fin"], cs] >= 0.6]
    cambios.append(int((prn != prn.shift()).sum() - 1) if len(prn) else 0)
t3 = pd.DataFrame([dict(eventos=len(evs), eventos_con_cambio_de_satelite=int(np.sum(np.array(cambios) > 0)),
                        cambios_medios_por_evento=round(float(np.mean(cambios)), 2) if cambios else None)])
out = [guardar(t1, "V06_elevacion_en_eventos.csv", "Elevación del satélite del máximo (S4 >= 0.6)"),
       guardar(t2, "V06_resumen_elevacion.csv", "Resumen"), guardar(t3, "V06_cambios_satelite.csv", "Cambios de satélite en eventos (k = 0)")]
bitacora("V06", "-", out)