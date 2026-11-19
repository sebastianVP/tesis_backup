"""V19b · Tabla 1.5 (particiones del conjunto con precursores) para la ejecución final y la repetición v6.
Columnas de la tesis: periodo, minutos, días, días con evento (%), rachas y minutos con S4 ≥ 0.6.
La fila «final» debe reproducir la Tabla 1.5 actual (entrenamiento: 428 515 minutos, 309 días, 64 (20.7 %), 361, 667).
Uso: python validacion/V19b_tabla_1_5.py"""
from comun import *
from s4.config import cargar_config
from s4.pipeline import preparar_datos
cabecera("V19b", "config/tesis_final.yaml + config/tesis_v6.yaml", "· solo conteos de datos")
def rachas(s):
    e = (s >= 0.6).astype(int).to_numpy(); t = s.index
    return int(((e == 1) & ((np.r_[0, e[:-1]] == 0) | (np.r_[np.inf, np.diff(t.asi8) / 6e10] > 1))).sum())
filas = []
for nombre, ruta in (("final", "config/tesis_final.yaml"), ("v6", "config/tesis_v6.yaml")):
    cfg = cargar_config(ruta); hc = cfg["split"].get("hora_corte_ut", 12)
    d = preparar_datos(cfg, usar_precursores=True, verbose=False)
    tot = sum(len(d[k]) for k in ("train", "val", "test"))
    for part in ("train", "val", "test"):
        s = d[part]["S4"]; dia = (s.index - pd.Timedelta(hours=hc)).normalize()
        de = pd.Series(s.to_numpy() >= 0.6, index=dia).groupby(level=0).any()
        filas.append(dict(ejecucion=nombre, particion=part, desde=str(s.index.min())[:16], hasta=str(s.index.max())[:16],
                          minutos=len(s), pct_minutos=round(100 * len(s) / tot, 1), dias=int(de.size), dias_con_evento=int(de.sum()),
                          pct_dias_evento=round(100 * de.mean(), 1), rachas=rachas(s), minutos_S4_ge_06=int((s >= 0.6).sum())))
out = [guardar(pd.DataFrame(filas), "V19b_tabla_1_5.csv", "Tabla 1.5 · ejecución final frente a repetición v6")]
bitacora("V19b", "config/tesis_final.yaml + config/tesis_v6.yaml", out)
