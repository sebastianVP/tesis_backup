"""V11 · Meseta de β con la configuración FINAL (C1-025), solo en VALIDACIÓN.
S4 + hora, L = 60, α = 1.5, β ∈ {5, 10, 25, 50}, 3 semillas, umbral calibrado.
Comprueba si la meseta 5–25 hallada con D' y L = 180 se mantiene en la configuración del v3.0.
Uso: python validacion/V11_beta_configuracion_final.py   (12 entrenamientos)"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); a = ap.parse_args()
cabecera("V11", a.config, "· solo validación")
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante
from s4.registry import leer, resumir_semillas
from s4.train import configurar_gpu
from s4 import features as ft
configurar_gpu()
cfg = variante(cargar_config(a.config), **{"nombre": "validacion", "ventanas.lookback": 60, "perdida.alpha": 1.5})
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
for b in (5, 10, 25, 50):
    correr_variante(variante(cfg, **{"perdida.beta": float(b)}), d, ft.CONJUNTOS["B_S4_ciclica"], f"b{b}", "V11_beta_final", barrido_umbral=True)
reg = leer(ruta(cfg, "experimentos") / "registro_validacion.csv", grupo="V11_beta_final", particion="val")
t = resumir_semillas(reg, por=("modelo",), metricas=("HSS_opt", "umbral_opt", "POD_opt", "Precision_opt", "POD_inicio_opt", "RMSE_evento"))
out = [guardar(t.round(4), "V11_beta_final.csv", "β con S4 + hora y L = 60 (validación)")]
bitacora("V11", a.config, out, "registro: experiments/registro_validacion.csv, grupo V11_beta_final")
