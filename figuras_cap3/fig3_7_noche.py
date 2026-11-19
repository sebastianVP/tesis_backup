"""Figura 3.7 · Caso de estudio de la noche del 6 de marzo de 2026 (prueba) con el modelo v3.0 (C3-010).

Alineación temporal (explícita): cada pronóstico se dibuja en el INSTANTE PRONOSTICADO. La curva «+10 min» en el
instante τ es el valor que el modelo pronosticó para τ diez minutos antes (emitido en t = τ − 10); la de «+20 min»,
el emitido en t = τ − 20. La franja de alerta se dibuja en el INSTANTE DE EMISIÓN t (alerta = máximo pronosticado
de los 20 minutos siguientes ≥ umbral del v3.0, 0.55). Mismo estilo claro que las demás figuras.
Uso (desde la raíz del proyecto, en s4_env):  python figuras_cap3/fig3_7_noche.py [--v30 RUTA] [--dia 2026-03-06]
Salida: Figura_3_7_noche_v6_3.png y .pdf, y una línea con la hora de la primera alerta y del primer S4 ≥ 0.6."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "validacion")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comun import CFG_DEFECTO, V30_DEFECTO, cargar_modelo_app
from s4.config import cargar_config
from s4.pipeline import preparar_datos
from s4.windows import preparar_particion, inicios_comunes
from s4.train import predecir
ap = argparse.ArgumentParser(); ap.add_argument("--v30", default=V30_DEFECTO); ap.add_argument("--dia", default="2026-03-06")
ap.add_argument("--desde", default="00:00"); ap.add_argument("--hasta", default="09:30"); a = ap.parse_args()
cfg = cargar_config(CFG_DEFECTO); H = cfg["ventanas"]["horizon"]; hc = cfg["split"].get("hora_corte_ut", 12)
d = preparar_datos(cfg, usar_precursores=False, verbose=False)
modelo, esc, conf = cargar_modelo_app(a.v30); L, U = conf["LOOKBACK"], conf["UMBRAL_ALERTA"]
st = inicios_comunes(d["test"].index, [30, 60, 360], H)[L]
p = preparar_particion("test", d["test"], esc, conf["FEATURES_COLS"], L, H, "S4", 1, hc, st)
yh = predecir(modelo, p, esc)
t = pd.DatetimeIndex(p.t)                                     # instante de emisión (último minuto observado)
t0, t1 = pd.Timestamp(f"{a.dia} {a.desde}"), pd.Timestamp(f"{a.dia} {a.hasta}")
obs = d["test"]["S4"].loc[t0:t1]
p10 = pd.Series(yh[:, 9], index=t + pd.Timedelta("10min")).loc[t0:t1]
p20 = pd.Series(yh[:, 19], index=t + pd.Timedelta("20min")).loc[t0:t1]
alerta = pd.Series(yh.max(axis=1) >= U, index=t).loc[t0:t1]

plt.rcParams.update({"figure.dpi": 300, "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 4.2))
ini = None
for ti, v in list(alerta.items()) + [(alerta.index[-1] + pd.Timedelta("1min"), False)]:
    if v and ini is None: ini = ti
    if not v and ini is not None:
        ax.axvspan(ini, ti, color="#F5B7B1", alpha=0.45, lw=0, zorder=0); ini = None
ax.plot(obs.index, obs.values, color="#222222", lw=0.9, label="S4 observado", zorder=3)
ax.plot(p10.index, p10.values, color="#8E44AD", lw=1.6, label="Pronóstico +10 min (dibujado en el instante pronosticado)", zorder=4)
ax.plot(p20.index, p20.values, color="#2E86C1", lw=1.6, label="Pronóstico +20 min (dibujado en el instante pronosticado)", zorder=4)
ax.axhline(0.6, color="#C0392B", ls="--", lw=1.1, label="Umbral de centelleo severo (0.6)", zorder=2)
ax.fill_between([], [], color="#F5B7B1", alpha=0.45, label=f"Alerta emitida (instante de emisión; umbral {U})")
ax.set_xlim(t0, t1); ax.set_ylim(0, max(1.0, float(obs.max()) + 0.05))
ax.set_xlabel(f"Hora UT · {a.dia} (prueba, no usada en el entrenamiento)"); ax.set_ylabel("Índice S4")
ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8.3, frameon=True, framealpha=0.92)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"Figura_3_7_noche_v6_3.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
pa = alerta[alerta].index.min() if alerta.any() else None
ps = obs[obs >= 0.6].index.min() if (obs >= 0.6).any() else None
print(f"Figura_3_7_noche_v6_3.png y .pdf generadas · primera alerta: {pa} · primer S4 ≥ 0.6: {ps}")
