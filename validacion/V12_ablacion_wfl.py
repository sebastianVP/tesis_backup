"""V12 · Ablación de los componentes de la WFL (C1-024, C1-011), solo en VALIDACIÓN.
S4 + hora, L = 60, 3 semillas, umbral calibrado. Variantes:
  MSE · WFL completa (α = 1.5, β = 10, γ = 2) · sin término focal (α = 0) ·
  sin penalización de subestimación (γ = 1) · sin peso de severidad (β = 1).
Reporta además el sesgo medio (ŷ − y) en los minutos con S4 ≥ 0.6.
Uso: python validacion/V12_ablacion_wfl.py   (15 entrenamientos)"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO); a = ap.parse_args()
cabecera("V12", a.config, "· solo validación")
from s4.config import cargar_config, variante
from s4.pipeline import preparar_datos, correr_variante
from s4.train import configurar_gpu, predecir
from s4 import features as ft
configurar_gpu()
base = variante(cargar_config(a.config), **{"nombre": "validacion", "ventanas.lookback": 60})
d = preparar_datos(base, usar_precursores=False, verbose=False)
VAR = [("MSE", {"perdida.tipo": "mse"}),
       ("WFL completa (α1.5 β10 γ2)", {"perdida.alpha": 1.5, "perdida.beta": 10.0, "perdida.penal_subestimacion": 2.0}),
       ("sin término focal (α0)", {"perdida.alpha": 0.0, "perdida.beta": 10.0, "perdida.penal_subestimacion": 2.0}),
       ("sin penalización (γ1)", {"perdida.alpha": 1.5, "perdida.beta": 10.0, "perdida.penal_subestimacion": 1.0}),
       ("sin peso de severidad (β1)", {"perdida.alpha": 1.5, "perdida.beta": 1.0, "perdida.penal_subestimacion": 2.0})]
filas = []
for nom, ov in VAR:
    cfg = variante(base, **ov)
    fs, mods, parts = correr_variante(cfg, d, ft.CONJUNTOS["B_S4_ciclica"], nom, "V12_ablacion_wfl", barrido_umbral=True, guardar_modelo=True)
    p = parts["val"]; m = ~p.objetivo_reconstruido
    for (sem, modelo, _), f in zip(mods, fs):
        yh = predecir(modelo, p, parts["escalador"])[m]; y = p.y_real[m]; e = y >= 0.6
        filas.append(dict(variante=nom, semilla=sem, HSS_cal=f["HSS_opt"], umbral=f["umbral_opt"], POD_cal=f["POD_opt"],
                          Precision_cal=f["Precision_opt"], inicios_cal=f["POD_inicio_opt"],
                          sesgo_eventos=float((yh[e] - y[e]).mean()), RMSE_evento=f["RMSE_evento"], RMSE_global=f["RMSE_global"]))
    del mods
T = pd.DataFrame(filas)
R = T.drop(columns="semilla").groupby("variante").agg(["mean", "std"]).round(4); R.columns = [f"{x}_{y}" for x, y in R.columns]
out = [guardar(T.round(4), "V12_ablacion_wfl_semillas.csv"), guardar(R.reset_index(), "V12_ablacion_wfl.csv", "Ablación de la WFL (validación)")]
bitacora("V12", a.config, out, "registro: experiments/registro_validacion.csv, grupo V12_ablacion_wfl")
