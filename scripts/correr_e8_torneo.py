"""
E8 · TORNEO DE ARQUITECTURAS CON EL PROTOCOLO FINAL (experimento complementario)

Repite el torneo E1 (LSTM simple, apilada y bidireccional) con el protocolo corregido:
partición Out-of-Time fija, solo S4 + seno/coseno de la hora y del día del año,
WFL α = 1.5 y β = 10 (la del modelo principal), 3 semillas y umbral de alerta
calibrado en VALIDACIÓN. Compara además tres periodos de entrenamiento; validación
(10/12/2025 – 02/03/2026) y prueba no cambian, porque los cortes están fijos en el YAML.

    2025       : 26/01/2025 – 09/12/2025  (referencia: el modelo principal)
    2024-2025  : 01/02/2024 – 09/12/2025  (régimen homogéneo de tres constelaciones)
    2022-2025  : 01/01/2022 – 09/12/2025  (serie completa, mezcla de regímenes)

REGLA (fijada antes de ver resultados): una combinación solo se considera mejor que la
referencia «apilado · 2025» si su HSS calibrado medio la supera en más de 0.03 Y las tres
semillas mejoran a su semilla par. NO se evalúa en prueba: este script solo usa validación.

Nota: la variante bidireccional del paquete incluye BatchNormalization tras cada capa,
como en el torneo original.

Uso (desde la raíz del proyecto):
    nohup python scripts/correr_e8_torneo.py > experiments/log_e8_torneo.txt 2>&1 &
    python scripts/correr_e8_torneo.py --solo-resumen          # rehace la tabla con lo ya corrido
Opciones: --arquitecturas simple apilado bidireccional · --periodos 2025 2024-2025 2022-2025
Si se interrumpe, al relanzarlo salta las combinaciones (modelo, semilla) ya registradas.
"""
import argparse, sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import pandas as pd
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante
from s4.registry import leer, resumir_semillas
from s4 import features as ft

PERIODOS = {"2025": "2025-01-26", "2024-2025": "2024-02-01", "2022-2025": "2022-01-01"}
GRUPO, REF = "E8_torneo", "apilado · 2025"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/tesis_final.yaml")
ap.add_argument("--arquitecturas", nargs="+", default=["simple", "apilado", "bidireccional"])
ap.add_argument("--periodos", nargs="+", default=list(PERIODOS))
ap.add_argument("--solo-resumen", action="store_true")
a = ap.parse_args()

base = variante(cargar_config(a.config), **{"perdida.alpha": 1.5, "perdida.beta": 10.0, "ventanas.lookback": 60})
REG = ruta(base, "experimentos") / f"registro_{base['nombre']}.csv"
FEATS = ft.CONJUNTOS["B_S4_ciclica"]
print(f"E8 · variables: {FEATS} · ventana 60 → 20 min · WFL α = 1.5, β = 10 · registro: {REG}", flush=True)

def hechos():
    try:
        r = leer(REG, grupo=GRUPO, particion="val")
        return set(zip(r["modelo"], r["semilla"].astype(int)))
    except (FileNotFoundError, KeyError):
        return set()

if not a.solo_resumen:
    from s4.train import configurar_gpu
    configurar_gpu()
    for etiqueta in a.periodos:
        cfg_p = variante(base, **{"periodo.inicio": PERIODOS.get(etiqueta, etiqueta)})
        print(f"\n######## Periodo de entrenamiento {etiqueta} (desde {cfg_p['periodo']['inicio']}) ########", flush=True)
        datos = preparar_datos(cfg_p, usar_precursores=False)
        tr = datos["train"]
        print(f"  entrenamiento: {tr.index.min()} → {tr.index.max()} | {len(tr):,} minutos | "
              f"{tr.index.normalize().nunique()} días", flush=True)
        for arq in a.arquitecturas:
            nombre = f"{arq} · {etiqueta}"
            pendientes = [s for s in base["entrenamiento"]["semillas"] if (nombre, s) not in hechos()]
            if not pendientes:
                print(f"  {nombre}: ya registrado, se omite", flush=True); continue
            print(f"\n===== {nombre} · semillas {pendientes} =====", flush=True)
            correr_variante(variante(cfg_p, **{"modelo.tipo": arq}), datos, FEATS, nombre, GRUPO,
                            semillas=pendientes, barrido_umbral=True)

# ---------- resumen y regla de decisión ----------
r = leer(REG, grupo=GRUPO, particion="val").copy()
r["arquitectura"] = r["modelo"].str.split(" · ").str[0]
r["periodo_entrenamiento"] = r["modelo"].str.split(" · ").str[1]
met = ("HSS_opt", "umbral_opt", "Precision_opt", "POD_inicio_opt", "RMSE_evento", "RMSE_global", "epocas", "seg_entrenamiento")
t = resumir_semillas(r, por=("arquitectura", "periodo_entrenamiento"), metricas=met)
ref = r[r["modelo"] == REF].set_index("semilla")["HSS_opt"]
filas = []
for (arq, per), g in r.groupby(["arquitectura", "periodo_entrenamiento"]):
    g = g.set_index("semilla")["HSS_opt"]
    comunes = g.index.intersection(ref.index)
    dif = g.mean() - ref.mean() if len(ref) else float("nan")
    todas = bool(len(comunes) == 3 and (g[comunes] > ref[comunes]).all())
    filas.append(dict(arquitectura=arq, periodo_entrenamiento=per, n_semillas=len(g),
                      dif_HSS_vs_ref=round(dif, 3), tres_semillas_mejoran=todas,
                      supera_regla=bool(dif > 0.03 and todas) if f"{arq} · {per}" != REF else None))
t = t.merge(pd.DataFrame(filas), on=["arquitectura", "periodo_entrenamiento"])
orden = {"2025": 0, "2024-2025": 1, "2022-2025": 2}
t = t.sort_values(["periodo_entrenamiento", "arquitectura"], key=lambda s: s.map(orden) if s.name == "periodo_entrenamiento" else s)
salida = ruta(base, "reportes") / "tables" / "E8_torneo.csv"
salida.parent.mkdir(parents=True, exist_ok=True); t.to_csv(salida, index=False)
pd.set_option("display.width", 250)
print("\n=== E8_torneo (validación) ===\n" + t.round(3).to_string(index=False))
ganan = t[t["supera_regla"] == True]
print("\nDECISIÓN: " + ("ninguna combinación supera a la referencia según la regla; se mantiene apilado · 2025."
                        if ganan.empty else "superan la regla: " + ", ".join(ganan["arquitectura"] + " · " + ganan["periodo_entrenamiento"])))
print(f"Tabla guardada en {salida}\nE8 TERMINADO", flush=True)