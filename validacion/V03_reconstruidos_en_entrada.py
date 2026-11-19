"""V03 · Minutos reconstruidos en la ENTRADA X de las ventanas (C1-016).
El código actual solo marca si algún minuto del OBJETIVO Y fue reconstruido.
Este script cuenta, por partición y para L = 60 y 180 (conjunto sin precursores):
ventanas con algún minuto reconstruido en X, en Y y en X ∪ Y.
Uso: python validacion/V03_reconstruidos_en_entrada.py"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); a = ap.parse_args()
cabecera("V03", a.config)
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import inicios_validos
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
filas = []
for k in ("train", "val", "test"):
    p = d[k]; rec = p["reconstruido"].to_numpy(bool)
    pref = np.r_[0, np.cumsum(rec)]
    for L in (60, 180):
        st = inicios_validos(p.index, L, H)
        en_x = (pref[st + L] - pref[st]) > 0
        en_y = (pref[st + L + H] - pref[st + L]) > 0
        filas.append(dict(particion=k, L=L, ventanas=len(st), con_rec_en_X=int(en_x.sum()),
                          pct_rec_en_X=round(100 * en_x.mean(), 2), con_rec_en_Y=int(en_y.sum()),
                          pct_rec_en_Y=round(100 * en_y.mean(), 2), pct_rec_en_X_o_Y=round(100 * (en_x | en_y).mean(), 2)))
out = [guardar(pd.DataFrame(filas), "V03_reconstruidos_en_X.csv", "Ventanas con minutos reconstruidos")]
bitacora("V03", a.config, out, "Las métricas del v3.0 sin estas ventanas se calculan en V08")
