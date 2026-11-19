"""V19 · Tabla 3.1 (particiones, días, rachas y ventanas con evento, ventana de 180 min) para la ejecución
final (config/tesis_final.yaml) y para la repetición v6 (config/tesis_v6.yaml, retención de 15 min).
La primera debe reproducir la Tabla 3.1 actual (426 326 / 95 616 / 165 009 ventanas): si coincide, las
definiciones son las mismas y la segunda da los valores v6.
Uso: python validacion/V19_tabla_3_1.py"""
import argparse
from comun import *
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import inicios_validos
ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=180); a = ap.parse_args()
cabecera("V19", "config/tesis_final.yaml + config/tesis_v6.yaml", "· solo conteos de datos")

def rachas(s):
    e = (s >= 0.6).astype(int).to_numpy(); t = s.index
    nuevo = (e == 1) & ((np.r_[0, e[:-1]] == 0) | (np.r_[np.inf, np.diff(t.asi8) / 6e10] > 1))
    return int(nuevo.sum())

filas = []
for nombre, ruta in (("final", "config/tesis_final.yaml"), ("v6", "config/tesis_v6.yaml")):
    cfg = cargar_config(ruta); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
    d = preparar_datos(cfg, usar_precursores=True, verbose=False)
    for part in ("train", "val", "test"):
        df = d[part]; s = df["S4"]
        dia = (df.index - pd.Timedelta(hours=hc)).normalize()
        dias_evento = pd.Series(s.to_numpy() >= 0.6, index=dia).groupby(level=0).any()
        st = inicios_validos(df.index, a.L, H)
        y = s.to_numpy()[st[:, None] + a.L + np.arange(H)[None, :]]
        ev = (y >= 0.6).any(axis=1)
        filas.append(dict(ejecucion=nombre, particion=part, desde=str(df.index.min())[:16], hasta=str(df.index.max())[:16],
                          dias=int(dias_evento.size), dias_con_evento=int(dias_evento.sum()), rachas=rachas(s),
                          ventanas=int(len(st)), con_evento=int(ev.sum()), pct_con_evento=round(100 * ev.mean(), 2)))
T = pd.DataFrame(filas)
out = [guardar(T, "V19_tabla_3_1.csv", f"Tabla 3.1 · ejecución final frente a repetición v6 (ventana de {a.L} min)")]
print("\nTabla 3.1 actual (final): 309 (64) días, 361 rachas, 426 326 ventanas, 3 897 con evento (0.91 %) · validación 73 (16), 116, 95 616, 1 075 (1.12 %) · prueba 125 (13), 253, 165 009, 2 251 (1.36 %)")
bitacora("V19", "config/tesis_final.yaml + config/tesis_v6.yaml", out)
