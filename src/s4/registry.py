"""Registro de experimentos: una fila por (configuración, semilla, partición).

Reemplaza el registro_experimentos.csv anterior, que mezclaba corridas viejas y
nuevas con filas sin nombre. Cada fila lleva la huella de la configuración:
si dos filas tienen la misma huella y semilla, deben ser reproducibles.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .config import huella


def registrar(ruta_csv, cfg: dict, fila: dict, grupo: str, semilla=None, extra: dict | None = None) -> dict:
    ruta_csv = Path(ruta_csv)
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    registro = dict(fecha=dt.datetime.now().isoformat(timespec="seconds"), grupo=grupo,
                    huella=huella(cfg), semilla=semilla, **(extra or {}), **fila,
                    config=json.dumps({k: cfg.get(k) for k in ("ventanas", "perdida", "modelo", "features")},
                                      default=str))
    nueva = pd.DataFrame([registro])
    if ruta_csv.exists():
        cabecera = list(pd.read_csv(ruta_csv, nrows=0).columns)
        if set(nueva.columns) <= set(cabecera):
            # misma cabecera: se agrega la fila con las columnas en el MISMO orden
            nueva.reindex(columns=cabecera).to_csv(ruta_csv, mode="a", header=False, index=False)
            return registro
        # aparecen columnas nuevas: se reescribe el archivo con la unión de columnas
        previo = pd.read_csv(ruta_csv)
        pd.concat([previo, nueva], ignore_index=True).to_csv(ruta_csv, index=False)
        return registro
    nueva.to_csv(ruta_csv, index=False)
    return registro


def leer(ruta_csv, grupo=None, particion=None) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv)
    if grupo is not None:
        df = df[df["grupo"] == grupo]
    if particion is not None:
        df = df[df["particion"] == particion]
    return df


def resumir_semillas(df: pd.DataFrame, por=("modelo",), metricas=("RMSE_global", "RMSE_evento", "HSS", "POD", "Precision")):
    """media ± desviación entre semillas (lo que va a las tablas de la tesis)."""
    metricas = [m for m in metricas if m in df.columns]
    g = df.groupby(list(por))[metricas].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()
