"""V05 · Escala de θ y ρ, y pesos efectivos de la WFL (C1-023).
Lee el escalador del v3.0 y de la variante B, y calcula:
  · θ en escala del modelo y el umbral físico equivalente de la regla ŷ < ρ·y;
  · la contribución relativa de un minuto de evento frente a uno tranquilo
    (cada término ∝ w·p·|e|^(2+α)), evaluando la propia función de pérdida del paquete.
Uso: python validacion/V05_escalador_y_wfl.py [--v30 RUTA] [--b RUTA]"""
import argparse, joblib
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--b", default=B_DEFECTO); a = ap.parse_args()
cabecera("V05", "-")
filas = []
for nom, c in (("v3.0", a.v30), ("variante B", a.b)):
    p = Path(c) / "SCALER_S4_app.pkl"
    if not p.exists():
        print(f"(no existe {p})"); continue
    esc = joblib.load(p); conf = json.load(open(Path(c) / "config.json"))
    mn, mx = float(esc.data_min_[0]), float(esc.data_max_[0])
    th = (0.6 - mn) / (mx - mn)
    filas.append(dict(modelo=nom, S4_min_entrenamiento=mn, S4_max_entrenamiento=mx, theta_escala_modelo=round(th, 5),
                      regla_rho_en_escala_fisica=f"ŷ < 0.8·y + {0.2 * mn:.4f}", alpha=conf.get("ALPHA"), beta=conf.get("BETA")))
t1 = pd.DataFrame(filas)
import tensorflow as tf
from s4.losses import weighted_focal_mse
casos = []
for beta in (10.0, 100.0):
    for alpha in (1.5,):
        L = weighted_focal_mse(0.5, alpha, beta, 2.0, 0.8)
        def termino(y, yh):
            return float(L(tf.constant([[y]], tf.float32), tf.constant([[yh]], tf.float32)))
        calma = termino(0.10, 0.12)               # minuto tranquilo, |e| = 0.02
        ev_sobre = termino(0.70, 0.90)            # evento sobreestimado, |e| = 0.20 (sin penalización)
        ev_sub = termino(0.90, 0.70)              # evento subestimado, |e| = 0.20 (ŷ < 0.8·y)
        casos.append(dict(alpha=alpha, beta=beta, perdida_calma_e002=calma, perdida_evento_e020=ev_sobre,
                          perdida_evento_sub_e020=ev_sub, razon_evento_calma=round(ev_sobre / calma),
                          razon_evento_sub_calma=round(ev_sub / calma)))
out = [guardar(t1, "V05_escalador_theta_rho.csv", "Escala de θ y ρ"),
       guardar(pd.DataFrame(casos), "V05_pesos_wfl.csv", "Pesos efectivos de la WFL (escala del modelo, umbral 0.5)")]
bitacora("V05", "-", out, "exponente efectivo 2 + α; razones calculadas con la propia función del paquete")
