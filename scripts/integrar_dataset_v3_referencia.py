# ============================================================
# PIPELINE DE INTEGRACIÓN (v3, sin fuga de información, con S4 reconstruido)
#
# Integra, a resolución de 1 minuto:
#   1) S4 procesado (LISN, máximo por minuto)
#   2) TEC / ROTEC / ROTI (GloTEC, SWPC-NOAA; un mapa cada 10 min)
#   3) Índices OMNI (Kp, ap trihorarios; Dst, AE horarios; F10.7 diario)
#
# Salida:  integrated_dataset/DF_FINAL_<ESTACION>_<ddmmaaaa>.csv
#
# CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR
# ------------------------------------------------------------
# 1. SIN INTERPOLACIÓN LINEAL. La versión anterior interpolaba TEC
#    (cada 10 min) y los índices OMNI (cada 1 h, 3 h o 1 día) hacia
#    cada minuto con interpolate(method="time"): el minuto 10:35 ya
#    "conocía" el valor de las 11:00 (fuga de información de hasta
#    10 min en TEC, 1 h en Dst/AE, 3 h en Kp/ap y 1 día en F10.7).
#    Ahora cada valor se mantiene constante desde su marca de tiempo
#    hasta la siguiente (forward-fill con límite), sin usar el futuro.
# 2. HUECOS: el forward-fill tiene un límite por variable; si falta un
#    dato por más tiempo, el minuto queda vacío (NaN) en lugar de
#    inventarse con una recta.
# 3. DESPLAZAMIENTO CAUSAL DE LOS ÍNDICES: los valores OMNI se dejan en
#    su hora de etiqueta. El paquete s4 (causalizar_indice) los desplaza
#    un periodo completo antes de entrenar. Si este CSV se usa fuera de
#    s4, activar DESPLAZAR_INDICES = True (¡no ambos a la vez!).
# 4. NO SE DESCARTAN FILAS POR PRECURSORES FALTANTES: solo se exige S4.
#    Se reporta la cobertura de cada columna.
# 5. TEC: acepta la columna de tiempo "datetime" (scripts v2) o
#    "Datetime"; verifica que la celda GloTEC sea la de la estación.
# 6. JICAMARCA usa el archivo TEC de Jicamarca (misma celda que Lima).
# 7. (v3) El S4 de entrada es el RECONSTRUIDO (df_max_s4_reconstruido.csv) con sus
#    marcas Fix0–Fix4, de modo que el archivo integrado es la ÚNICA fuente de datos
#    del paquete s4: variable objetivo, marcas de reconstrucción y precursores.
#    Las columnas ID_Satelite/Azimuth/Elevacion se reemplazan por Constelacion y
#    N_Sats (vacías en los minutos reconstruidos).
# ============================================================

from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================
# RUTAS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

S4_DIR = PROJECT_DIR / "S4_LISN" / "reconstruido"          # df_max_s4_reconstruido.csv
TEC_DIR = PROJECT_DIR / "TEC" / "TEC_ROTI_DPTO"
OMNI_DIR = PROJECT_DIR / "OMNIWEB_NASA" / "OMNI_DATA_CSV"
OUTPUT_DIR = SCRIPT_DIR / "integrated_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# PARÁMETROS
# ============================================================

# Máximo de minutos que se mantiene un valor si falta el siguiente
LIMITE_FFILL_MIN = {
    "TEC": 15, "ROTEC": 15, "ROTI": 15,            # mapas cada 10 min
    "Dst_Index": 60, "AE_Index": 60,               # horarios
    "Kp_Index": 180, "ap_Index": 180,              # trihorarios
    "f10.7_Index": 1440,                           # diario
}

# Solo para usar el CSV FUERA del paquete s4 (que ya desplaza los índices):
DESPLAZAR_INDICES = False
PERIODO_INDICE = {"Dst_Index": "1h", "AE_Index": "1h", "Kp_Index": "3h", "ap_Index": "3h", "f10.7_Index": "1D"}

ESTACIONES = {
    "JICAMARCA": {"s4": "jic", "tec": "Jicamarca", "latlon": (-11.95, -76.87)},
    "HUANCAYO": {"s4": "hyo", "tec": "Huancayo", "latlon": (-12.04, -75.32)},
    "PIURA": {"s4": "piu", "tec": "Piura", "latlon": (-5.17, -80.64)},
    "CUZCO": {"s4": "cuz", "tec": "Cusco", "latlon": (-13.52, -71.97)},
    "PUCALLPA": {"s4": "pucall", "tec": "Pucallpa", "latlon": (-8.38, -74.54)},
    "AYACUCHO": {"s4": "aya", "tec": "Ayacucho", "latlon": (-13.16, -74.22)},
    "TACNA": {"s4": "tac", "tec": "Tacna", "latlon": (-18.01, -70.25)},
    "IQUITOS": {"s4": "iqui", "tec": "Iquitos", "latlon": (-3.75, -73.25)},
}

FLAGS = ["Fix0", "Fix1", "Fix3", "Fix4", "Fix_Pico"]
COLUMNAS_FINALES = [
    "S4", "Fix0", "Fix1", "Fix3", "Fix4", "Fix_Pico", "Constelacion", "N_Sats",
    "TEC", "ROTEC", "ROTI",
    "Kp_Index", "Dst_Index", "ap_Index", "f10.7_Index", "AE_Index",
]

# ============================================================
# FUNCIONES
# ============================================================

def leer_con_tiempo(ruta):
    """Lee un CSV con columna de tiempo 'Datetime', 'datetime' o índice sin nombre (UTC, sin zona)."""
    df = pd.read_csv(ruta)
    col = next(c for c in ["Datetime", "datetime", "Tiempo", "Unnamed: 0"] if c in df.columns)
    df[col] = pd.to_datetime(df[col])
    if getattr(df[col].dt, "tz", None) is not None:
        df[col] = df[col].dt.tz_localize(None)
    return df.set_index(col).sort_index()


def a_minuto_causal(df, malla, limites):
    """Lleva cada columna a la malla de 1 minuto SIN interpolar: el valor de
    una marca de tiempo se mantiene hasta la siguiente (máx. `limites[col]` min)."""
    df = df[~df.index.duplicated(keep="first")]
    union = malla.union(df.index)
    salida = pd.DataFrame(index=malla)
    for col in df.columns:
        salida[col] = df[col].reindex(union).ffill(limit=limites.get(col, 0)).reindex(malla)
    return salida


def desplazar_indices(df_omni):
    """Desplaza cada índice un periodo completo (solo si se usa fuera de s4)."""
    salida = {}
    for col, periodo in PERIODO_INDICE.items():
        s = df_omni[col].dropna()
        nodos = s.groupby(s.index.floor(periodo)).first()      # valor del periodo
        nodos.index = nodos.index + pd.Timedelta(periodo)       # disponible al terminar el periodo
        salida[col] = nodos
    return pd.DataFrame(salida)


def integrar_datasets(file_s4, file_tec, file_omni, latlon):
    print("\n=== LECTURA ===")
    df_s4 = leer_con_tiempo(file_s4)
    df_s4 = df_s4[~df_s4.index.duplicated(keep="first")]
    for f in FLAGS:                                   # marcas de reconstrucción como booleanos
        df_s4[f] = df_s4[f].astype(str).str.lower().isin(["true", "1"]) if f in df_s4.columns else False
    print(f"S4 reconstruido: {len(df_s4):,} minutos | reconstruidos: "
          f"{int(df_s4[['Fix1', 'Fix3', 'Fix4']].any(axis=1).sum()):,} | S4 NaN: {int(df_s4['S4'].isna().sum())}")
    df_tec = leer_con_tiempo(file_tec)
    df_omni = leer_con_tiempo(file_omni)

    # Verificación de la celda GloTEC
    if {"lat", "lon"} <= set(df_tec.columns):
        lat_c, lon_c = df_tec["lat"].iloc[0], df_tec["lon"].iloc[0]
        print(f"Celda GloTEC: ({lat_c}, {lon_c}) | estación: {latlon}")
        if abs(lat_c - latlon[0]) > 2.5 or abs(lon_c - latlon[1]) > 5:
            raise ValueError("La celda GloTEC no corresponde a la estación")
    df_tec = df_tec[["TEC", "ROTEC", "ROTI"]]

    indices = ["Kp_Index", "Dst_Index", "ap_Index", "f10.7_Index", "AE_Index"]
    df_omni = df_omni[indices]
    if DESPLAZAR_INDICES:
        print("Índices OMNI desplazados un periodo completo (uso fuera de s4)")
        df_omni = desplazar_indices(df_omni)

    # Periodo común y malla de 1 minuto
    inicio = max(df_s4.index.min(), df_tec.index.min(), df_omni.index.min())
    fin = min(df_s4.index.max(), df_tec.index.max(), df_omni.index.max())
    print(f"Periodo común: {inicio} → {fin}")
    malla = pd.date_range(inicio.ceil("min"), fin.floor("min"), freq="1min")

    tec_min = a_minuto_causal(df_tec, malla, LIMITE_FFILL_MIN)
    omni_min = a_minuto_causal(df_omni, malla, LIMITE_FFILL_MIN)

    cols_s4 = ["S4"] + FLAGS + [c for c in ["Constelacion", "N_Sats"] if c in df_s4.columns]
    df = pd.concat([df_s4[cols_s4].reindex(malla), tec_min, omni_min], axis=1)
    df = df.dropna(subset=["S4"])                              # solo se exige S4
    for c in ["Constelacion", "N_Sats"]:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLUMNAS_FINALES].reset_index().rename(columns={"index": "Tiempo"})

    print("\nCobertura de cada columna (% de minutos con dato):")
    print((100 * df[COLUMNAS_FINALES].notna().mean()).round(2).to_string())
    return df


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    N_ESTACION = 0
    estacion = list(ESTACIONES)[N_ESTACION]
    conf = ESTACIONES[estacion]

    FILE_S4 = S4_DIR / "df_max_s4_reconstruido.csv"            # salida del notebook de reconstrucción
    FILE_TEC = TEC_DIR / f"TEC_ROTI_{conf['tec']}.csv"
    FILE_OMNI = OMNI_DIR / "omni_processed.csv"
    for f in [FILE_S4, FILE_TEC, FILE_OMNI]:
        if not f.exists():
            raise FileNotFoundError(f"No existe el archivo:\n{f}")
    print(f"Estación: {estacion}\nS4  : {FILE_S4}\nTEC : {FILE_TEC}\nOMNI: {FILE_OMNI}")

    df_final = integrar_datasets(FILE_S4, FILE_TEC, FILE_OMNI, conf["latlon"])
    df_final.insert(1, "Estacion", estacion)

    # ROTEC y ROTI son del orden de 1e-3 a 1e-5: se guardan con 8 decimales
    flotantes = df_final.select_dtypes(include=["float64", "float32"]).columns
    df_final[flotantes] = df_final[flotantes].round(8)

    salida = OUTPUT_DIR / f"DF_FINAL_{estacion}_integrado_{datetime.now():%d%m%Y}.csv"
    df_final.to_csv(salida, index=False)
    print(f"\nRegistros: {len(df_final):,} | Columnas: {len(df_final.columns)}\nArchivo: {salida}")
