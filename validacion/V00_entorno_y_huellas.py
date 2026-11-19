"""V00 · Entorno, commit y huellas (sha256) de datos, configuraciones, modelos y registros.
Responde: reproducibilidad (hash de commit y de cada archivo usado por la tesis).
Uso: python validacion/V00_entorno_y_huellas.py [--v30 RUTA] [--b RUTA]"""
import argparse, platform, sys
from comun import *

ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--b", default=B_DEFECTO); a = ap.parse_args()
cabecera("V00", a.config)
from s4.config import cargar_config, ruta
cfg = cargar_config(a.config)
vers = {"python": platform.python_version()}
for m in ("numpy", "pandas", "sklearn", "tensorflow", "keras", "statsmodels"):
    try:
        vers[m] = __import__(m).__version__
    except Exception as e:
        vers[m] = f"no disponible ({type(e).__name__})"
try:
    import tensorflow as tf
    vers["gpu"] = ", ".join(g.name for g in tf.config.list_physical_devices("GPU")) or "ninguna"
except Exception:
    vers["gpu"] = "?"
vers["commit"] = commit()
t0 = pd.DataFrame([vers])
archivos = [ruta(cfg, "s4_reconstruido"), ruta(cfg, "multivariable"), Path(a.config), Path("config/tesis_final.yaml"),
            ruta(cfg, "experimentos") / f"registro_{cfg['nombre']}.csv"]
for carpeta in (a.v30, a.b):
    for f in ("config.json", "SCALER_S4_app.pkl", "modelo_S4_lstm.keras", "app.py"):
        archivos.append(Path(carpeta) / f)
filas = []
for f in archivos:
    f = Path(f)
    filas.append(dict(archivo=str(f), existe=f.exists(), bytes=f.stat().st_size if f.exists() else None,
                      sha256_16=sha256(f) if f.exists() else None))
t1 = pd.DataFrame(filas)
reg = ruta(cfg, "experimentos") / f"registro_{cfg['nombre']}.csv"
t2 = pd.read_csv(reg).groupby(["grupo", "particion"]).size().rename("filas").reset_index() if reg.exists() else pd.DataFrame()
for carpeta, nom in ((a.v30, "v3.0"), (a.b, "variante B")):
    p = Path(carpeta) / "config.json"
    if p.exists():
        c = json.load(open(p))
        print(f"\n{nom}: " + ", ".join(f"{k}={c.get(k)}" for k in ("LOOKBACK", "HORIZON", "ALPHA", "BETA", "UMBRAL_ALERTA", "SEMILLA", "FEATURES_COLS") if k in c))
out = [guardar(t0, "V00_entorno.csv", "Entorno"), guardar(t1, "V00_huellas_archivos.csv", "Huellas de archivos"),
       guardar(t2, "V00_registro_grupos.csv", "Grupos del registro")]
if vers["commit"] == "SIN-GIT":
    print("\n⚠ El proyecto no está bajo git. Ejecute antes:  git init && git add -A && git commit -m 'v5.3 auditada'")
bitacora("V00", a.config, out)
