"""
Comparación del MODELO FINAL (ya evaluado en prueba) con los modelos anteriores v2.0 y v3.0
sobre EXACTAMENTE los mismos instantes objetivo de la prueba.

No reentrena ni recalibra nada: carga los artefactos exportados de cada modelo
(app/modelo_final_tesis, v3.0 y v2.0) y calcula sus métricas sobre los instantes
comunes (los que tienen historia completa para la ventana más larga, 360 min).
Por eso las cifras del modelo final pueden diferir levemente de la tabla de E7,
que usaba todos los instantes con 30 min de historia.

Uso:
    python scripts/comparar_referencias.py --v20 <carpeta_v2.0> --v30 <carpeta_v3.0>
Cada carpeta debe contener config.json, SCALER_S4_app.pkl y modelo_S4_lstm.keras.
"""
import argparse, json, sys, warnings
from pathlib import Path
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from tensorflow.keras.models import load_model
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, evaluar_y_registrar
from s4.windows import inicios_comunes, preparar_particion
from s4.train import configurar_gpu, predecir
from s4 import stats, evaluate as ev

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/tesis_final.yaml")
ap.add_argument("--final", default="app/modelo_final_tesis")
ap.add_argument("--v30", required=True); ap.add_argument("--v20", required=True)
a = ap.parse_args()
configurar_gpu()
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]
datos = preparar_datos(cfg, usar_precursores=False)
MODELOS = [("Modelo final", Path(a.final)), ("Modelo v3.0", Path(a.v30)), ("Modelo v2.0", Path(a.v20))]
conf = {}
for n, p in MODELOS:
    for f in ["config.json", "SCALER_S4_app.pkl", "modelo_S4_lstm.keras"]:
        assert (p / f).exists(), f"Falta {p / f}"
    conf[n] = json.load(open(p / "config.json"))
    print(f"{n:13s} · {p} · ventana {conf[n]['LOOKBACK']} min · umbral {conf[n].get('UMBRAL_ALERTA')} · "
          f"α={conf[n].get('ALPHA')} β={conf[n].get('BETA')}", flush=True)
LBS = sorted({c["LOOKBACK"] for c in conf.values()})
com = inicios_comunes(datos["test"].index, LBS, H)

preds, parts = {}, {}
for n, p in MODELOS:
    c = conf[n]; esc = joblib.load(p / "SCALER_S4_app.pkl"); esc.columnas_ = c["FEATURES_COLS"]
    mod = load_model(p / "modelo_S4_lstm.keras", compile=False)
    parts[n] = preparar_particion(f"test_{n}", datos["test"], esc, c["FEATURES_COLS"], c["LOOKBACK"], H,
                                  hora_corte_ut=cfg["split"].get("hora_corte_ut", 12), starts=com[c["LOOKBACK"]])
    preds[n] = (predecir(mod, parts[n], esc), float(c.get("UMBRAL_ALERTA", 0.65)))
pt = parts["Modelo final"]
for n in parts:
    assert (parts[n].t == pt.t).all(), f"Los instantes objetivo no coinciden ({n})"
preds["Persistencia"] = (np.repeat(pt.ultimo_real[:, None], H, axis=1), 0.6)
print(f"\nInstantes objetivo comunes: {len(pt.t):,} | con evento: {int((pt.y_real.max(axis=1) >= 0.6).sum()):,}", flush=True)

filas = [evaluar_y_registrar(cfg, pt, yh, n, "final_test_comun", umbral_alerta=u) for n, (yh, u) in preds.items()]
tab = pd.DataFrame(filas); TAB = ruta(cfg, "reportes") / "tables"
tab.to_csv(TAB / "tabla_final_test_comun.csv", index=False)
cols = ["modelo", "umbral_alerta", "RMSE_global", "RMSE_evento", "POD", "Precision", "FAR", "HSS", "POD_inicio", "antelacion_media_min"]
print("\n===== MISMOS INSTANTES DE PRUEBA =====\n" + tab[cols].round(4).to_string(index=False), flush=True)

m = ~pt.objetivo_reconstruido; y = pt.y_real
alinear = lambda yh, u: yh - u + 0.6
yf, uf = preds["Modelo final"]; sig = []
for ref in ["Modelo v3.0", "Modelo v2.0", "Persistencia"]:
    yr, ur = preds[ref]
    b_hss = stats.bootstrap_bloques(y, alinear(yr, ur), alinear(yf, uf), pt.dia, stats.m_menos_hss(0.6, 0.6), mascara=m)
    b_rmse = stats.bootstrap_bloques(y, yr, yf, pt.dia, stats.m_rmse_evento(), mascara=m)
    dm = stats.diebold_mariano(y, yr, yf, pt.dia, mascara=m)
    sig.append(dict(referencia=ref, mejora_HSS=b_hss["diferencia"], IC95_HSS=b_hss["ic95"], sig_HSS=b_hss["significativo"],
                    mejora_RMSE_evento=b_rmse["diferencia"], IC95_RMSE_evento=b_rmse["ic95"], sig_RMSE=b_rmse["significativo"],
                    DM=dm["DM"], p_DM=dm["p_valor"]))
sig = pd.DataFrame(sig); sig.to_csv(TAB / "tabla_final_significancia_comun.csv", index=False)
print("\n===== SIGNIFICANCIA: modelo final frente a cada referencia (positivo = mejora del modelo final) =====\n"
      + sig.round(4).to_string(index=False), flush=True)
fases = pd.concat({n: ev.metricas_por_fase(pt, yh, umbral_alerta=u) for n, (yh, u) in preds.items()}, names=["modelo"]).reset_index(level=0)
fases.to_csv(TAB / "tabla_final_fases_comun.csv", index=False)
print("\n===== POR FASE =====\n" + fases.round(3).to_string(index=False))
print("\nCOMPARACIÓN TERMINADA", flush=True)
