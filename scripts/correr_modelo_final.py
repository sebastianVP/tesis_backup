"""Notebook 09 · Modelo final: única evaluación en TEST y exportación a la aplicación.

Configuración elegida en VALIDACIÓN (ejecución final, reglas pre-registradas):
  LSTM apilada 64/64 · S4 + hora (B_S4_ciclica) · ventana 30 min · horizonte 20 min · WFL α = 1.0, β = 5
  (E2 + punto de control: ventana · E4: α y β · E5: arquitectura · E6: variables).

Pasos:
  1. Reentrenar con 3 semillas; elegir la semilla con mayor HSS calibrado en VALIDACIÓN y fijar su umbral.
  2. Evaluar UNA SOLA VEZ en TEST frente a persistencia, climatología y los modelos anteriores
     (v2.0: 360 min; v3.0: 60 min), sobre EXACTAMENTE los mismos instantes objetivo.
  2b. Verificación complementaria en temporada de calma (22/07 – 30/09/2026, datos nunca usados):
     alertas emitidas por el modelo final (casi no hay eventos: solo se cuentan falsas alarmas).
  3. Significancia: bootstrap por días (HSS y RMSE en eventos) y Diebold–Mariano.
  4. Exportar modelo, escalador y config.json a app/modelo_final/.
"""
import json, os, shutil, sys, warnings
from pathlib import Path
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante, baselines, evaluar_y_registrar
from s4.train import configurar_gpu, predecir, medir_latencia
from s4.windows import inicios_comunes, preparar_particion
from s4 import features as ft, evaluate as ev, stats

LB = 30
REFERENCIAS = [("Modelo v2.0", Path(os.environ.get("S4_V20", "app/produccion_actual"))),
               ("Modelo v3.0", Path(os.environ.get("S4_V30", "respaldo_v1/modelo_final_v3_desplegado")))]
REFERENCIAS = [(n, p) for n, p in REFERENCIAS if (p / "config.json").exists()]
PROSPECTIVO = ("2026-07-22", "2026-09-30 23:59")
configurar_gpu()
cfg = variante(cargar_config(os.environ.get("S4_CFG", "config/tesis_final.yaml")),
               **{"ventanas.lookback": LB, "perdida.alpha": 1.0, "perdida.beta": 5.0})
H = cfg["ventanas"]["horizon"]
TAB, FIG = ruta(cfg, "reportes") / "tables", ruta(cfg, "reportes") / "figures"
print(f"MODELO FINAL · LOOKBACK={LB} · alpha=1.0 · beta=5 · S4 + hora", flush=True)
print("Referencias encontradas:", [n for n, _ in REFERENCIAS], flush=True)
datos = preparar_datos(cfg, usar_precursores=False)      # el modelo final solo usa S4 y la hora
FEATS = ft.CONJUNTOS["B_S4_ciclica"]

# ---- objetivos comunes en TEST (para comparar con el modelo desplegado de 360 min) ----
CONF_REF = {n: json.load(open(p / "config.json")) for n, p in REFERENCIAS}
LBS = sorted({LB, *[c["LOOKBACK"] for c in CONF_REF.values()]})
com = inicios_comunes(datos["test"].index, LBS, H)

# ---- 1. Entrenamiento con 3 semillas y selección en VALIDACIÓN ----
filas, modelos, parts = correr_variante(cfg, datos, FEATS, "Modelo final", "final_val",
                                        barrido_umbral=True, guardar_modelo=True,
                                        starts={"test": com[LB]})
mejor = int(np.argmax([f["HSS_opt"] for f in filas]))
semilla, modelo, _ = modelos[mejor]
U = float(filas[mejor]["umbral_opt"])
esc = parts["escalador"]
print(f"\nSemilla elegida en validación: {semilla} | HSS calibrado {filas[mejor]['HSS_opt']:.3f} | umbral {U}", flush=True)

# ---- 2. ÚNICA evaluación en TEST ----
pt = parts["test"]
y = pt.y_real
preds = {"Modelo final": (predecir(modelo, pt, esc), U)}
for nombre, yh in baselines(cfg, datos, parts, "test").items():
    preds[nombre] = (yh, 0.6)
from tensorflow.keras.models import load_model
for nombre_ref, carpeta in REFERENCIAS:
    conf = CONF_REF[nombre_ref]
    esc_d = joblib.load(carpeta / "SCALER_S4_app.pkl"); esc_d.columnas_ = conf["FEATURES_COLS"]
    mod_d = load_model(carpeta / "modelo_S4_lstm.keras", compile=False)
    p_d = preparar_particion(f"test_{nombre_ref}", datos["test"], esc_d, conf["FEATURES_COLS"], conf["LOOKBACK"], H,
                             hora_corte_ut=cfg["split"].get("hora_corte_ut", 12), starts=com[conf["LOOKBACK"]])
    assert (p_d.t == pt.t).all(), f"Los instantes objetivo no coinciden ({nombre_ref})"
    preds[nombre_ref] = (predecir(mod_d, p_d, esc_d), float(conf.get("UMBRAL_ALERTA", 0.65)))
if not REFERENCIAS:
    print("(No se encontraron modelos anteriores: se omite esa comparación)", flush=True)

filas_t = [evaluar_y_registrar(cfg, pt, yh, n, "final_test", semilla if n == "Modelo final" else None, umbral_alerta=u)
           for n, (yh, u) in preds.items()]
final = pd.DataFrame(filas_t)
final.to_csv(TAB / "tabla_final_test.csv", index=False)
cols = ["modelo", "umbral_alerta", "RMSE_global", "RMSE_evento", "POD", "Precision", "FAR", "HSS", "POD_inicio", "antelacion_media_min"]
print("\n===== RESULTADOS EN TEST (única evaluación) =====")
print(final[cols].round(4).to_string(index=False), flush=True)

# ---- 3. Significancia (cada modelo con su propio umbral: se desplaza la predicción para alertar en 0.6) ----
m = ~pt.objetivo_reconstruido
alinear = lambda yh, u: yh - u + 0.6
sig = []
yf, uf = preds["Modelo final"]
for ref in [k for k in preds if k != "Modelo final"]:
    yr, ur = preds[ref]
    a = (y, yr, yf, pt.dia)
    b_hss = stats.bootstrap_bloques(y, alinear(yr, ur), alinear(yf, uf), pt.dia, stats.m_menos_hss(0.6, 0.6), mascara=m)
    b_rmse = stats.bootstrap_bloques(*a, stats.m_rmse_evento(), mascara=m)
    dm = stats.diebold_mariano(*a, mascara=m)
    sig.append(dict(referencia=ref, mejora_HSS=b_hss["diferencia"], IC95_HSS=b_hss["ic95"], sig_HSS=b_hss["significativo"],
                    mejora_RMSE_evento=b_rmse["diferencia"], IC95_RMSE_evento=b_rmse["ic95"], sig_RMSE=b_rmse["significativo"],
                    DM=dm["DM"], p_DM=dm["p_valor"]))
sig = pd.DataFrame(sig); sig.to_csv(TAB / "tabla_final_significancia.csv", index=False)
print("\n===== SIGNIFICANCIA: modelo final frente a cada referencia (positivo = mejora del modelo final) =====")
print(sig.round(4).to_string(index=False), flush=True)

fases = pd.concat({n: ev.metricas_por_fase(pt, yh, umbral_alerta=u) for n, (yh, u) in preds.items()}, names=["modelo"]).reset_index(level=0)
fases.to_csv(TAB / "tabla_final_fases.csv", index=False)
fa = ev.analisis_falsas_alarmas(pt, yf, umbral_alerta=uf); fa.to_csv(TAB / "tabla_final_falsas_alarmas.csv")
print("\n===== POR FASE =====\n", fases.round(3).to_string(index=False))
print("\n===== FALSAS ALARMAS DEL MODELO FINAL (S4 real máximo) =====\n", fa.to_string(), flush=True)

np.savez_compressed(TAB / "predicciones_test.npz", t=pt.t.values.astype("datetime64[m]").astype(np.int64),
                    y=y.astype(np.float32), mascara=m, **{n.replace(" ", "_").replace(".", ""): yh.astype(np.float32) for n, (yh, _) in preds.items()})
lat = medir_latencia(modelo, pt, n=200, dispositivo="/CPU:0")
print(f"\nLatencia en CPU: mediana {lat['lat_ms_mediana']:.1f} ms · parámetros {lat['parametros']:,}", flush=True)

# ---- 2b. Verificación en temporada de calma (datos posteriores a la prueba) ----
from s4 import io as s4io
serie = s4io.cargar_s4_reconstruido(ruta(cfg, "s4_reconstruido"))
ini_p, fin_p = PROSPECTIVO
dfp = ft.agregar_ciclicas(serie.loc[pd.Timestamp(ini_p) - pd.Timedelta(minutes=LB):fin_p])
if len(dfp) > LB + H:
    pp = preparar_particion("prospectivo", dfp, esc, FEATS, LB, H, hora_corte_ut=cfg["split"].get("hora_corte_ut", 12))
    pp_t = pd.DatetimeIndex(pp.t); dentro = pp_t >= pd.Timestamp(ini_p)
    yp = predecir(modelo, pp, esc)[dentro]; yr = pp.y_real[dentro]; tt = pp_t[dentro]
    alerta = yp.max(axis=1) >= U; evento = yr.max(axis=1) >= 0.6
    episodios = int((alerta & ~np.r_[False, alerta[:-1]]).sum())
    dias = len(np.unique(tt.normalize())); dias_alerta = len(np.unique(tt[alerta].normalize()))
    resumen_p = {"periodo": f"{ini_p} a {fin_p}", "dias": dias, "ventanas": int(len(alerta)),
                 "ventanas_con_evento": int(evento.sum()), "ventanas_con_alerta": int(alerta.sum()),
                 "falsas_alarmas": int((alerta & ~evento).sum()), "episodios_de_alerta": episodios,
                 "dias_con_alerta": dias_alerta, "episodios_por_dia": round(episodios / max(dias, 1), 3)}
    pd.Series(resumen_p).to_csv(TAB / "tabla_final_prospectivo_calma.csv")
    print("\n===== VERIFICACIÓN EN TEMPORADA DE CALMA (22/07 – 30/09/2026) =====")
    print(pd.Series(resumen_p).to_string(), flush=True)
else:
    print("(Sin datos posteriores a la prueba para la verificación de calma)", flush=True)

# ---- 4. Exportación para la aplicación ----
dest = Path(cfg["rutas"]["base"]) / "app" / os.environ.get("S4_DESTINO", "modelo_final_tesis"); dest.mkdir(parents=True, exist_ok=True)
modelo.save(dest / "modelo_S4_lstm.keras", include_optimizer=False)
joblib.dump(esc, dest / "SCALER_S4_app.pkl")
fila = final[final["modelo"] == "Modelo final"].iloc[0]
json.dump({"LOOKBACK": LB, "HORIZON": H, "THRESHOLD": cfg["datos"]["umbral_evento"], "UMBRAL_ALERTA": U,
           "FEATURES_COLS": list(esc.columnas_), "TARGET_COL": "S4", "SAMPLING_MINUTES": 1,
           "ARQUITECTURA": "apilado", "UNITS1": 64, "UNITS2": 64, "ALPHA": 1.0, "BETA": 5.0, "SEMILLA": int(semilla),
           "RMSE_EVENTO_TEST": round(float(fila["RMSE_evento"]), 4), "HSS_TEST": round(float(fila["HSS"]), 4),
           "POD_TEST": round(float(fila["POD"]), 4), "PRECISION_TEST": round(float(fila["Precision"]), 4),
           "POD_INICIO_TEST": round(float(fila["POD_inicio"]), 4), "PERIODO_TEST": f"{pt.t.min():%Y-%m-%d} a {pt.t.max():%Y-%m-%d}"},
          open(dest / "config.json", "w"), indent=2, ensure_ascii=False)
if Path("app/produccion_actual/app.py").exists():
    shutil.copy("app/produccion_actual/app.py", dest / "app.py")
print(f"\nArtefactos exportados en {dest}", flush=True)
print("\nMODELO FINAL TERMINADO", flush=True)
