"""Utilidades comunes del proyecto de validación de la tesis (scintia-s4).

Todos los scripts:
  · se ejecutan desde la raíz del proyecto:  python validacion/Vxx_....py [opciones]
  · escriben sus resultados en validacion/salidas/ (CSV) y una entrada en BITACORA.md
    con fecha, commit de git, script, configuración y archivos generados;
  · NUNCA usan el conjunto de prueba para tomar decisiones. Los scripts que lo leen
    (V08, V09) son re-análisis declarados del modelo ya elegido.
"""
from __future__ import annotations
import datetime as dt, hashlib, subprocess, sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, pandas as pd

RAIZ = Path.cwd()
SAL = RAIZ / "validacion" / "salidas"; SAL.mkdir(parents=True, exist_ok=True)
CFG_DEFECTO = "config/tesis_final.yaml"
V30_DEFECTO = str(Path.home() / "Documents" / "s4_lstm_env_test")          # app v3.0 desplegada
B_DEFECTO = "app/modelo_final_tesis"                                      # variante B (replicación)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)


def commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return (h + ("-modificado" if sucio else "")) if h else "SIN-GIT"
    except Exception:
        return "SIN-GIT"


def sha256(ruta, n=1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()[:16]


def cabecera(script, cfg_path, extra=""):
    print(f"=== {script} · {dt.datetime.now():%Y-%m-%d %H:%M} · commit {commit()} · config {cfg_path} {extra}", flush=True)


def guardar(df: pd.DataFrame, nombre: str, titulo: str | None = None) -> Path:
    p = SAL / nombre; df.to_csv(p, index=False)
    if titulo:
        print(f"\n--- {titulo} ---\n" + df.to_string(index=False), flush=True)
    return p


def bitacora(script: str, cfg_path: str, archivos: list, notas: str = ""):
    with open(SAL / "BITACORA.md", "a", encoding="utf8") as f:
        f.write(f"\n## {script} · {dt.datetime.now():%Y-%m-%d %H:%M}\n- commit: `{commit()}`\n- config: `{cfg_path}`\n")
        for a in archivos:
            f.write(f"- salida: `{Path(a).relative_to(RAIZ) if Path(a).is_absolute() else a}`\n")
        if notas:
            f.write(f"- notas: {notas}\n")
    print(f"\n[{script}] terminado · resultados en {SAL}", flush=True)


def eventos_unidos(s4: pd.Series, umbral=0.6, k_union=0) -> pd.DataFrame:
    """Rachas de minutos consecutivos con S4 >= umbral. Si k_union > 0, dos rachas
    separadas por <= k_union minutos (sin dato o bajo el umbral) se unen en un evento."""
    s = s4.dropna(); act = s[s >= umbral]
    if act.empty:
        return pd.DataFrame(columns=["inicio", "fin", "duracion_min", "minutos_sobre_umbral", "max_S4"])
    t = act.index; corte = np.r_[True, (t[1:] - t[:-1]) > pd.Timedelta(minutes=1 + k_union)]
    gid = np.cumsum(corte)
    g = pd.DataFrame({"t": t, "S4": act.values, "g": gid}).groupby("g")
    ev = g.agg(inicio=("t", "min"), fin=("t", "max"), minutos_sobre_umbral=("t", "size"), max_S4=("S4", "max"))
    ev["duracion_min"] = ((ev["fin"] - ev["inicio"]) / pd.Timedelta("1min")).astype(int) + 1
    return ev.reset_index(drop=True)[["inicio", "fin", "duracion_min", "minutos_sobre_umbral", "max_S4"]]


def cargar_modelo_app(carpeta):
    """Carga modelo, escalador y config.json de una carpeta exportada (v3.0 o variante B)."""
    import joblib
    from tensorflow.keras.models import load_model
    c = Path(carpeta)
    conf = json.load(open(c / "config.json"))
    esc = joblib.load(c / "SCALER_S4_app.pkl"); esc.columnas_ = conf["FEATURES_COLS"]
    return load_model(c / "modelo_S4_lstm.keras", compile=False), esc, conf
