"""Notebook 08: ¿qué variables usa realmente la red? (pregunta del especialista sobre ROTI)

Configuración elegida: D' · S4 + hora + ROTI · ventana 180 min · WFL α = 1.5, β = 10 · umbral 0.55.
1) Importancia por permutación AGRUPADA (3 semillas, VALIDACIÓN completa): se permuta un grupo
   de variables entre ventanas y se mide cuánto empeora el modelo. Las variables cíclicas se
   permutan en pares (seno y coseno juntos) para no crear horas imposibles.
2) Gradientes integrados en ventanas de INICIO de evento (semilla 42): cuánto aporta cada
   variable en cada minuto de la ventana de 3 horas.
"""
import os, sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante
from s4.train import configurar_gpu
from s4.windows import a_escala_real
from s4 import features as ft, evaluate as ev, explain

LB = int(os.environ.get("S4_LB", 180))
U_ALERTA = 0.55
configurar_gpu()
cfg = variante(cargar_config(os.environ.get("S4_CFG", "config/ablacion_2025_2026.yaml")),
               **{"ventanas.lookback": LB, "perdida.alpha": 1.5, "perdida.beta": 10.0})
print("LOOKBACK =", LB, "| umbral de alerta =", U_ALERTA, flush=True)
datos = preparar_datos(cfg)
FEATS = ft.CONJUNTOS["D_solo_ROTI"]
GRUPOS = {"S4": ["S4"], "Hora del día": ["daily_sin", "daily_cos"],
          "Día del año": ["day_of_year_sin", "day_of_year_cos"], "ROTI": ["ROTI"]}

filas, modelos, parts = correr_variante(cfg, datos, FEATS, "D' explicabilidad", "explicabilidad_D",
                                        barrido_umbral=True, guardar_modelo=True)
p, esc = parts["val"], parts["escalador"]
m = ~p.objetivo_reconstruido
X = p.X_lote(np.flatnonzero(m))                     # validación completa (sin objetivos reconstruidos)
y, ult = p.y_real[m], p.ultimo_real[m]
fase = ev.fase_ventana(ult, y)
print(f"Ventanas evaluadas: {len(X):,} | inicios de evento: {(fase == 'inicio').sum():,}", flush=True)

def metricas(yhat):
    a = ev.metricas_alerta(y, yhat, 0.6, U_ALERTA)
    ini = fase == "inicio"
    pod_ini = float((yhat[ini].max(axis=1) >= U_ALERTA).mean()) if ini.any() else np.nan
    ev_ = y >= 0.6
    return dict(HSS=a["HSS"], Precision=a["Precision"], POD_inicio=pod_ini,
                RMSE_evento=float(np.sqrt(np.mean((y[ev_] - yhat[ev_]) ** 2))),
                RMSE_global=float(np.sqrt(np.mean((y - yhat) ** 2))))

def predecir(modelo, Xm):
    return a_escala_real(modelo.predict(Xm, batch_size=2048, verbose=0), esc)

rng = np.random.default_rng(0)
imp = []
for semilla, modelo, _ in modelos:
    base = metricas(predecir(modelo, X))
    imp.append(dict(semilla=semilla, grupo="(sin permutar)", **base))
    for g, cols in GRUPOS.items():
        idx = [FEATS.index(c) for c in cols]
        for rep in range(3):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, :, idx] = X[perm][:, :, idx]       # mismo orden para todo el grupo
            mm = metricas(predecir(modelo, Xp))
            imp.append(dict(semilla=semilla, grupo=g, repeticion=rep, **mm))
            print(f"  semilla {semilla} | {g:14s} | rep {rep}: HSS {base['HSS']:.3f} -> {mm['HSS']:.3f}", flush=True)
        del Xp
imp = pd.DataFrame(imp)
TAB, FIG = ruta(cfg, "reportes") / "tables", ruta(cfg, "reportes") / "figures"
imp.to_csv(TAB / "tabla_importancia_permutacion_detalle.csv", index=False)

base = imp[imp["grupo"] == "(sin permutar)"].set_index("semilla")
d = imp[imp["grupo"] != "(sin permutar)"].copy()
for mtr in ["HSS", "Precision", "POD_inicio"]:
    d[f"caida_{mtr}"] = d["semilla"].map(base[mtr]) - d[mtr]          # cuánto se pierde
for mtr in ["RMSE_evento", "RMSE_global"]:
    d[f"aumento_{mtr}"] = d[mtr] - d["semilla"].map(base[mtr])        # cuánto empeora
res = d.groupby("grupo")[[c for c in d.columns if c.startswith(("caida_", "aumento_"))]].agg(["mean", "std"])
res.columns = [f"{a}_{b}" for a, b in res.columns]
res = res.sort_values("caida_HSS_mean", ascending=False).reset_index()
res.to_csv(TAB / "tabla_importancia_permutacion.csv", index=False)
print("\nIMPORTANCIA POR PERMUTACIÓN (media entre 3 semillas × 3 repeticiones)")
print(res.round(4).to_string(index=False), flush=True)

# ---------- Gradientes integrados en ventanas de inicio (semilla 42) ----------
modelo42 = [mo for s, mo, _ in modelos if s == 42][0] if any(s == 42 for s, _, _ in modelos) else modelos[0][1]
ini_idx = np.flatnonzero(fase == "inicio")
sel = rng.choice(ini_idx, min(400, len(ini_idx)), replace=False)
at = np.concatenate([explain.gradientes_integrados(modelo42, X[sel[i:i + 100]], pasos=32) for i in range(0, len(sel), 100)])
mapa = np.abs(at).mean(axis=0)                       # (LB, n_features)
sal = pd.DataFrame(mapa, columns=FEATS); sal.insert(0, "minutos_antes_de_t", np.arange(LB - 1, -1, -1))
sal.to_csv(TAB / "tabla_saliencia_inicio.csv", index=False)
tot = pd.Series({g: mapa[:, [FEATS.index(c) for c in cols]].sum() for g, cols in GRUPOS.items()})
print("\nGRADIENTES INTEGRADOS (ventanas de inicio): participación de cada grupo")
print((100 * tot / tot.sum()).round(1).to_string(), flush=True)
roti = mapa[:, FEATS.index("ROTI")]
tramos = {"últimos 30 min": roti[-30:].sum(), "30–60 min antes": roti[-60:-30].sum(),
          "1–2 h antes": roti[-120:-60].sum(), "2–3 h antes": roti[:-120].sum() if LB > 120 else 0.0}
print("\nROTI: dónde está su aporte dentro de la ventana")
print(pd.Series(tramos).div(roti.sum()).mul(100).round(1).to_string(), flush=True)
print("\nNOTEBOOK 08 TERMINADO", flush=True)
