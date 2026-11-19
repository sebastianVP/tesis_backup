"""V09 · Variabilidad entre semillas del v3.0 en PRUEBA (C1-026).
El v3.0 desplegado es una sola semilla (7); las otras dos no se guardaron. Este script
REENTRENA la configuración del v3.0 (S4 + hora, L = 60, α = 1.5, β = 10) con las semillas
42, 123 y 7, calibra el umbral de cada una en validación y las evalúa en prueba sobre los
mismos instantes de la Tabla 4.8. Es un análisis declarado: no reemplaza al v3.0.
Nota: por el no determinismo de la GPU, la semilla 7 reentrenada no es idéntica a la
desplegada; por eso se reporta también el v3.0 desplegado.
Uso: python validacion/V09_semillas_v30_en_prueba.py [--v30 RUTA]   (≈ 30–60 min en GPU)"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); ap.add_argument("--v30", default=V30_DEFECTO)
ap.add_argument("--determinista", action="store_true"); a = ap.parse_args()
cabecera("V09", a.config, "· PRUEBA: re-análisis declarado")
from s4.config import cargar_config, variante
from s4.pipeline import preparar_datos, correr_variante
from s4.windows import preparar_particion, inicios_comunes
from s4.train import configurar_gpu, predecir
from s4 import evaluate as ev, features as ft
configurar_gpu(determinista=a.determinista)
base = cargar_config(a.config)
cfg = variante(base, **{"nombre": "validacion", "ventanas.lookback": 60, "perdida.alpha": 1.5, "perdida.beta": 10.0})
d = preparar_datos(cfg, usar_precursores=False, verbose=False); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
com = inicios_comunes(d["test"].index, [30, 60, 360], H)
filas, mods, parts = correr_variante(cfg, d, ft.CONJUNTOS["B_S4_ciclica"], "v3.0 reentrenado", "V09_v30_semillas",
                                     barrido_umbral=True, guardar_modelo=True)
res = []
for (sem, modelo, _), f in zip(mods, filas):
    esc = parts["escalador"]
    pt = preparar_particion("test", d["test"], esc, ft.CONJUNTOS["B_S4_ciclica"], 60, H, "S4", 1, hc, com[60])
    yh = predecir(modelo, pt, esc); m = ~pt.objetivo_reconstruido
    for etiqueta, u in (("umbral calibrado en validación", f["umbral_opt"]), ("umbral 0.55 (v3.0)", 0.55)):
        a_ = ev.metricas_alerta(pt.y_real[m], yh[m], 0.6, u); ai = ev.anticipacion_inicio(pt, yh, 0.6, u)
        res.append(dict(modelo="reentrenado", semilla=sem, umbral_tipo=etiqueta, umbral=u, HSS_val_cal=f["HSS_opt"],
                        HSS_test=a_["HSS"], POD_test=a_["POD"], Precision_test=a_["Precision"], inicios_por_ventana=ai["POD_inicio"]))
if (Path(a.v30) / "config.json").exists():
    modelo, esc, conf = cargar_modelo_app(a.v30)
    pt = preparar_particion("test", d["test"], esc, conf["FEATURES_COLS"], 60, H, "S4", 1, hc, com[60]); yh = predecir(modelo, pt, esc)
    m = ~pt.objetivo_reconstruido; a_ = ev.metricas_alerta(pt.y_real[m], yh[m], 0.6, conf["UMBRAL_ALERTA"])
    res.append(dict(modelo="v3.0 desplegado", semilla=conf.get("SEMILLA"), umbral_tipo="umbral del v3.0", umbral=conf["UMBRAL_ALERTA"],
                    HSS_test=a_["HSS"], POD_test=a_["POD"], Precision_test=a_["Precision"],
                    inicios_por_ventana=ev.anticipacion_inicio(pt, yh, 0.6, conf["UMBRAL_ALERTA"])["POD_inicio"]))
T = pd.DataFrame(res)
R = T[T.modelo == "reentrenado"].groupby("umbral_tipo")[["HSS_test", "POD_test", "Precision_test", "inicios_por_ventana"]].agg(["mean", "std"]).round(4)
R.columns = [f"{x}_{y}" for x, y in R.columns]
out = [guardar(T.round(4), "V09_semillas_v30_prueba.csv", "Por semilla"), guardar(R.reset_index(), "V09_semillas_v30_resumen.csv", "Media ± desviación de 3 semillas")]
bitacora("V09", a.config, out, "registro: experiments/registro_validacion.csv, grupo V09_v30_semillas")
