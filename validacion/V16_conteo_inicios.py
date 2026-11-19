"""V16 · Verificación del conteo de inicios anticipados del v3.0 en prueba.
Reproduce paso a paso evaluate.anticipacion_inicio() sobre los mismos instantes de la Tabla 3.8
(comunes a L = 30, 60 y 360) y muestra cada número intermedio:
  ventanas de prueba → sin objetivo reconstruido → ventanas de INICIO (S4(t) < 0.6 y algún
  minuto del horizonte >= 0.6) → de ellas, con alerta (máximo pronosticado >= 0.55).
Guarda el detalle de cada ventana de inicio para revisarlo a mano.
Uso: python validacion/V16_conteo_inicios.py [--v30 RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); ap.add_argument("--v30", default=V30_DEFECTO); a = ap.parse_args()
cabecera("V16", a.config, "· PRUEBA: verificación de un conteo ya reportado")
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import preparar_particion, inicios_comunes
from s4.train import predecir
from s4 import evaluate as ev
cfg = cargar_config(a.config); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
modelo, esc, conf = cargar_modelo_app(a.v30); L, U = conf["LOOKBACK"], conf["UMBRAL_ALERTA"]
st = inicios_comunes(d["test"].index, [30, 60, 360], H)[L]
p = preparar_particion("test", d["test"], esc, conf["FEATURES_COLS"], L, H, "S4", 1, hc, st)
yh = predecir(modelo, p, esc)

m = ~p.objetivo_reconstruido
y, yhm, ult, t = p.y_real[m], yh[m], p.ultimo_real[m], pd.DatetimeIndex(p.t)[m]
fase = ev.fase_ventana(ult, y, 0.6)
ini = fase == "inicio"
alerta = yhm.max(axis=1) >= U
primer = np.argmax(y >= 0.6, axis=1) + 1
pasos = pd.DataFrame([
    dict(paso="1. Ventanas de prueba (instantes comunes L = 30, 60, 360)", valor=len(p.t)),
    dict(paso="2. Sin minutos reconstruidos en el objetivo", valor=int(m.sum())),
    dict(paso="3. Ventanas de INICIO: S4(t) < 0.6 y algún minuto de t+1..t+20 >= 0.6", valor=int(ini.sum())),
    dict(paso=f"4. Ventanas de inicio con alerta (máximo pronosticado >= {U})", valor=int((ini & alerta).sum())),
    dict(paso="5. Fracción = paso 4 / paso 3", valor=round(float((ini & alerta).sum() / ini.sum()), 6)),
    dict(paso="6. Antelación media (min hasta el primer minuto >= 0.6, solo con alerta)", valor=round(float(primer[ini & alerta].mean()), 2)),
    dict(paso="7. Instantes t distintos entre las ventanas de inicio", valor=int(pd.Index(t[ini]).nunique())),
    dict(paso="8. Noches (día ionosférico) con al menos una ventana de inicio", valor=int(pd.Index(pd.Series(p.dia)[m][ini]).nunique()))])
oficial = ev.anticipacion_inicio(p, yh, 0.6, U)
pasos.loc[len(pasos)] = dict(paso="Control · evaluate.anticipacion_inicio(): n_inicio / POD_inicio / antelación",
                             valor=f"{oficial['n_inicio']} / {oficial['POD_inicio']:.6f} / {oficial['antelacion_media_min']:.2f}")
det = pd.DataFrame(dict(t=t[ini], S4_t=ult[ini].round(3), max_real_20=y[ini].max(axis=1).round(3),
                        minuto_primer_06=primer[ini], max_pronosticado=yhm[ini].max(axis=1).round(3), alerta=alerta[ini]))
out = [guardar(pasos, "V16_conteo_inicios.csv", "Conteo de inicios anticipados del v3.0 (prueba)"),
       guardar(det, "V16_ventanas_inicio_detalle.csv")]
print("\nCómo leerlo: el paso 4 es el número de VENTANAS de inicio con alerta (no de episodios);"
      " cada inicio de evento genera hasta 20 ventanas (t = 20, 19, …, 1 minuto antes).")
bitacora("V16", a.config, out)
