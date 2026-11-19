"""V20 · Control de la ventana de historia con la configuración del v3.0, solo en VALIDACIÓN.

Pregunta: ¿es adecuada la ventana de 60 min del v3.0, o 30 o 120 min la superan con su misma configuración?
(E2 usó β = 100 y umbral fijo; el punto de control de 3.1.9 comparó solo 30 y 60 min con la configuración de
la variante B. Ninguno probó 120 min con la configuración del v3.0.)

Configuración (la del v3.0): S4 + hora (5 variables), LSTM apilada 64/64, WFL α = 1.5, β = 10, γ = 2, ρ = 0.8,
semillas 42, 123 y 7, umbral de alerta calibrado en validación. Ventanas L ∈ {30, 60, 120}, evaluadas sobre los
MISMOS instantes de validación (comunes a las tres ventanas). 9 entrenamientos.

REGLA DE DECISIÓN (fijada antes de ejecutar; hacer commit de este archivo antes de correrlo):
  Una ventana L ≠ 60 «supera» a 60 min solo si
    (a) su HSS calibrado medio supera al de 60 min en más de 0.03, y
    (b) sus tres semillas superan el HSS calibrado medio de 60 min.
  Si ninguna lo cumple → se confirma la ventana de 60 min del v3.0.
  Si alguna lo cumple → se reporta como limitación y candidata para la evaluación prospectiva;
  el modelo desplegado NO se cambia y la prueba NO se usa.

Uso (desde la raíz del proyecto, en s4_env):
  git add validacion/V20_control_ventana_v30.py && git commit -m "V20: regla fijada antes de ejecutar"
  python validacion/V20_control_ventana_v30.py
Salidas: validacion/salidas/V20_ventana_v30_por_semilla.csv y V20_ventana_v30_resumen.csv;
registro: experiments/registro_validacion.csv, grupo V20_ventana_v30."""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--ventanas", default="30,60,120"); a = ap.parse_args()
L_LIST = [int(v) for v in a.ventanas.split(",")]; REF, MARGEN = 60, 0.03
assert REF in L_LIST, "La ventana de referencia (60 min) debe estar en la lista"
cabecera("V20", a.config, "· solo validación · regla: > +0.03 de HSS calibrado medio y las 3 semillas sobre la media de 60 min")
from s4.config import cargar_config, variante
from s4.pipeline import preparar_datos, correr_variante
from s4.windows import inicios_comunes
from s4.train import configurar_gpu
from s4 import features as ft
configurar_gpu()
base = variante(cargar_config(a.config), **{"nombre": "validacion", "perdida.alpha": 1.5, "perdida.beta": 10.0})
H = base["ventanas"]["horizon"]
d = preparar_datos(base, usar_precursores=False, verbose=False)          # mismo conjunto que el v3.0
com = {k: inicios_comunes(d[k].index, L_LIST, H) for k in ("train", "val", "test")}
print(f"Instantes comunes de validación: {len(com['val'][REF]):,}")
filas = []
for L in L_LIST:
    cfg = variante(base, **{"ventanas.lookback": L})
    r = correr_variante(cfg, d, ft.CONJUNTOS["B_S4_ciclica"], f"LB{L}", "V20_ventana_v30",
                        starts={k: com[k][L] for k in com}, barrido_umbral=True)
    for sem, f in zip(cfg["entrenamiento"]["semillas"], r):
        filas.append(dict(ventana=L, semilla=sem, HSS_cal=f["HSS_opt"], umbral=f["umbral_opt"],
                          POD=f["POD_opt"], Precision=f["Precision_opt"], Inicios=f["POD_inicio_opt"],
                          RMSE_evento=f["RMSE_evento"], epocas=f.get("epocas")))
S = pd.DataFrame(filas)
ref = S[S.ventana == REF]["HSS_cal"].mean()
res = []
for L, g in S.groupby("ventana"):
    dif = g["HSS_cal"].mean() - ref
    tres = bool((g["HSS_cal"] > ref).all()) if L != REF else None
    supera = (dif > MARGEN and tres) if L != REF else None
    res.append(dict(ventana=L, HSS_cal_media=round(g["HSS_cal"].mean(), 4), HSS_cal_desv=round(g["HSS_cal"].std(), 4),
                    umbral_medio=round(g["umbral"].mean(), 2), RMSE_evento_media=round(g["RMSE_evento"].mean(), 4),
                    dif_vs_60=round(dif, 4), tres_semillas_sobre_media_60=tres, supera_regla=supera))
R = pd.DataFrame(res)
out = [guardar(S.round(4), "V20_ventana_v30_por_semilla.csv", "Por semilla (validación, instantes comunes)"),
       guardar(R, "V20_ventana_v30_resumen.csv", "Resumen y regla de decisión")]
ganan = R[R.supera_regla == True]["ventana"].tolist()
print("\nDECISIÓN: " + ("se CONFIRMA la ventana de 60 min del v3.0 (ninguna alternativa cumple la regla)." if not ganan else
      f"la(s) ventana(s) {ganan} cumplen la regla → reportar como limitación y candidata para la evaluación prospectiva; el v3.0 no se cambia."))
bitacora("V20", a.config, out, "registro: experiments/registro_validacion.csv, grupo V20_ventana_v30")
