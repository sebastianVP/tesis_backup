"""Repara un registro de experimentos con filas desalineadas.

Causa: cuando una partición no tenía ventanas de inicio de evento, la evaluación
devolvía 1 clave en lugar de 3 (n_inicio sin POD_inicio ni antelacion_media_min),
y las corridas de baselines no tienen las 3 columnas de entrenamiento. Las filas
se escribían sin cabecera propia, con distinto número de campos, y pandas las leía
corridas de columna: los filtros por 'grupo' devolvían tablas vacías.

Cada fila se reconstruye según su número de campos (el orden de las claves es fijo).
Uso:  python scripts/reparar_registro.py experiments/registro_ablacion_2025_2026.csv
"""
import csv, shutil, sys
import pandas as pd

BASE = ["fecha", "grupo", "huella", "semilla"]
EXTRA = ["epocas", "seg_entrenamiento", "features"]
FILA = (["modelo", "particion", "n_ventanas", "umbral_alerta",
         "RMSE_global", "MAE_global", "RMSE_evento", "MAE_evento", "RMSE_calma", "sesgo_evento", "RMSE_tH", "n_min_evento",
         "TP", "FP", "FN", "TN", "POD", "Precision", "FAR", "POFD", "CSI", "HSS", "F1"])
INI_CORTO, INI_LARGO = ["n_inicio"], ["n_inicio", "POD_inicio", "antelacion_media_min"]
ESQUEMAS = {len(BASE + FILA + INI_CORTO) + 1: BASE + FILA + INI_CORTO + ["config"],
            len(BASE + FILA + INI_LARGO) + 1: BASE + FILA + INI_LARGO + ["config"],
            len(BASE + EXTRA + FILA + INI_CORTO) + 1: BASE + EXTRA + FILA + INI_CORTO + ["config"],
            len(BASE + EXTRA + FILA + INI_LARGO) + 1: BASE + EXTRA + FILA + INI_LARGO + ["config"]}

ruta = sys.argv[1] if len(sys.argv) > 1 else "experiments/registro_ablacion_2025_2026.csv"
shutil.copy(ruta, ruta + ".respaldo")
filas, omitidas = [], 0
with open(ruta, newline="", encoding="utf-8") as f:
    lector = csv.reader(f)
    next(lector, None)                                   # cabecera original (no confiable)
    for campos in lector:
        esquema = ESQUEMAS.get(len(campos))
        if esquema is None:
            omitidas += 1
            continue
        filas.append(dict(zip(esquema, campos)))
df = pd.DataFrame(filas)
columnas = BASE + EXTRA + FILA + INI_LARGO + ["config"]
df = df.reindex(columns=columnas)
df.to_csv(ruta, index=False)
print(f"Filas reparadas: {len(df)} | omitidas: {omitidas} | respaldo: {ruta}.respaldo")
print(df.groupby(["grupo", "particion"]).size())
