"""V18 · Verificación de dos cifras de la sección 3.1.8 (v3.0 en prueba):
  (a) «el 81 % de las falsas alarmas ocurre con un S4 real máximo entre 0.4 y 0.6»
  (b) «emite alerta en […] el 89 % de las fases sostenidas y el 0.2 % de los minutos de calma»
Mismos instantes y máscara que la Tabla 3.8 y V16 (comunes a L = 30, 60, 360; sin objetivos reconstruidos).
Uso (desde la raíz del proyecto): python validacion/V18_falsas_alarmas_fases.py [--v30 RUTA]"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); ap.add_argument("--v30", default=V30_DEFECTO); a = ap.parse_args()
cabecera("V18", a.config, "· PRUEBA: verificación de cifras ya reportadas")
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
y, yhm, ult = p.y_real[m], yh[m], p.ultimo_real[m]
evento = y.max(axis=1) >= 0.6
alerta = yhm.max(axis=1) >= U
print(f"Ventanas evaluadas: {m.sum():,} · con evento: {evento.sum():,} · umbral de alerta: {U}")

# (a) Falsas alarmas: alerta sin evento → ¿qué S4 real máximo hubo en los 20 minutos?
fa = alerta & ~evento
mx = y[fa].max(axis=1)
tramos = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)]
filas_a = [dict(tramo_S4_max_real=f"[{lo:.1f}, {hi:.1f})", falsas_alarmas=int(((mx >= lo) & (mx < hi)).sum()),
                porcentaje=round(100 * ((mx >= lo) & (mx < hi)).mean(), 1)) for lo, hi in tramos]
filas_a.append(dict(tramo_S4_max_real="total", falsas_alarmas=int(fa.sum()), porcentaje=100.0))
A = pd.DataFrame(filas_a)

# (b) Tasa de alerta por fase de la ventana
fase = ev.fase_ventana(ult, y, 0.6)
B = (pd.DataFrame({"fase": fase, "alerta": alerta}).groupby("fase")["alerta"]
     .agg(ventanas="size", con_alerta="sum").reset_index())
B["porcentaje_con_alerta"] = (100 * B["con_alerta"] / B["ventanas"]).round(2)
B["POFD_si_calma"] = np.where(B["fase"] == "calma", (B["con_alerta"] / B["ventanas"]).round(4), np.nan)

out = [guardar(A, "V18_falsas_alarmas_por_tramo.csv", "(a) Falsas alarmas del v3.0 según el S4 real máximo del horizonte"),
       guardar(B, "V18_alerta_por_fase.csv", "(b) Tasa de alerta del v3.0 por fase de la ventana")]
print("\nCómo leerlo: (a) la fila [0.4, 0.6) debe dar ≈ 81 %; (b) la fase «sostenido» ≈ 89 %, «inicio» ≈ 32.5 % (587/1 804) y «calma» ≈ 0.2 %.")
bitacora("V18", a.config, out)
