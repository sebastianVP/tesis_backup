"""V02 · Particiones, definición de evento y desbalance (C1-020, C1-021, Resumen "1 %").
Para el conjunto SIN precursores (el del v3.0) y CON precursores (el de E2–E6):
  · minutos, proporción, días y noches con evento por partición;
  · eventos con distintas definiciones: rachas (k = 0, la de la tesis) y rachas
    unidas si las separan <= k minutos (k = 5, 10, 15, 30), con su duración;
  · % de ventanas con evento en el horizonte para L = 60 y L = 180.
Uso: python validacion/V02_particiones_y_eventos.py"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); a = ap.parse_args()
cabecera("V02", a.config)
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import inicios_validos
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]
filas, fe, fv = [], [], []
for etiqueta, usar in (("sin precursores (v3.0, variante B, E8)", False), ("con precursores (E2–E6)", True)):
    d = preparar_datos(cfg, usar_precursores=usar, verbose=False)
    total = sum(len(d[k]) for k in ("train", "val", "test"))
    for k in ("train", "val", "test"):
        p = d[k]; s = p["S4"]
        noche = s.resample("24h", offset="12h").max()
        filas.append(dict(conjunto=etiqueta, particion=k, inicio=p.index.min(), fin=p.index.max(), minutos=len(p),
                          pct_minutos=round(100 * len(p) / total, 1), noches=int(noche.notna().sum()),
                          noches_con_evento=int((noche >= 0.6).sum()), minutos_S4_ge_06=int((s >= 0.6).sum())))
        for kk in (0, 5, 10, 15, 30):
            ev = eventos_unidos(s, 0.6, kk)
            fe.append(dict(conjunto=etiqueta, particion=k, k_union_min=kk, eventos=len(ev),
                           duracion_media_min=round(ev["duracion_min"].mean(), 1) if len(ev) else None,
                           duracion_mediana_min=float(ev["duracion_min"].median()) if len(ev) else None,
                           duracion_p90_min=float(ev["duracion_min"].quantile(.9)) if len(ev) else None))
        for L in (60, 180):
            st = inicios_validos(p.index, L, H)
            y = s.to_numpy()[st[:, None] + L + np.arange(H)[None, :]]
            fv.append(dict(conjunto=etiqueta, particion=k, L=L, ventanas=len(st),
                           ventanas_con_evento=int((y.max(axis=1) >= .6).sum()),
                           pct_ventanas_con_evento=round(100 * (y.max(axis=1) >= .6).mean(), 2)))
out = [guardar(pd.DataFrame(filas), "V02_particiones.csv", "Particiones"),
       guardar(pd.DataFrame(fe), "V02_eventos_por_definicion.csv", "Eventos según la definición (k = minutos de unión)"),
       guardar(pd.DataFrame(fv), "V02_desbalance_ventanas.csv", "Proporción de ventanas con evento")]
bitacora("V02", a.config, out, "k = 0 reproduce la columna 'Eventos' de la tesis (rachas sin unión)")
