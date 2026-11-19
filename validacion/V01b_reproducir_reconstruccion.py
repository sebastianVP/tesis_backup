"""V01b · Reproduce la reconstrucción del S4 (notebook PRONOSTICO_S4_documentado) y la compara
con data/processed/df_max_s4_reconstruido.csv (C1-017).
Pasos idénticos al notebook: carga de A2 (df_max_s4_all_jic.csv) · cierre al 30/09/2026 ·
Fix0 (AuditorEventosS4 + corregir_picos_aislados) · ejecutar_pipeline_s4(max_gap=G).
ATENCIÓN: el notebook llamó ejecutar_pipeline_s4(df_fix0, max_gap=180), mientras la tesis dice
que los huecos de más de 60 min no se reconstruyen. Este script corre G = 180 y G = 60 y dice
cuál reproduce el archivo usado en los experimentos (semilla del remuestreo: 42).
Uso: python validacion/V01b_reproducir_reconstruccion.py [--a2 RUTA] [--gaps 180 60]   (≈ 20–60 min)"""
import argparse
from comun import *
ap = argparse.ArgumentParser(); ap.add_argument("--config", default=CFG_DEFECTO)
ap.add_argument("--a2", default=str(Path.home() / "Documents/pipelines_cip/S4_LISN/processed_s4/df_max_s4_all_jic.csv"))
ap.add_argument("--cierre", default="2026-09-30 23:59:59"); ap.add_argument("--gaps", type=int, nargs="+", default=[180, 60]); a = ap.parse_args()
cabecera("V01b", a.config)
from s4.config import cargar_config, ruta
from s4.reconstruction import CargadorDatosS4, AuditorEventosS4, ConfigAuditoria, corregir_picos_aislados, ejecutar_pipeline_s4
cfg = cargar_config(a.config)
ref = pd.read_csv(ruta(cfg, "s4_reconstruido"), index_col=0, parse_dates=True).loc[:a.cierre]
df = CargadorDatosS4(a.a2, montar_drive=False).cargar().loc[:a.cierre]
print(f"A2 cerrado: {len(df):,} minutos · {df.index.min()} → {df.index.max()}")
epis, _ = AuditorEventosS4(ConfigAuditoria()).auditar(df)
df_fix0, bit = corregir_picos_aislados(df, epis, modo="media_local", flag_col="Fix0")
print(f"Fix0: {int(df_fix0['Fix0'].sum())} minutos corregidos")
filas = []
for G in a.gaps:
    r = ejecutar_pipeline_s4(df_fix0, max_gap=G)["df4"]
    comun = r.index.intersection(ref.index)
    dif = (r.loc[comun, "S4"].astype(float) - ref.loc[comun, "S4"].astype(float)).abs()
    fila = dict(max_gap=G, filas_reproducidas=len(r), filas_archivo=len(ref), filas_en_comun=len(comun),
                solo_reproducidas=len(r.index.difference(ref.index)), solo_archivo=len(ref.index.difference(r.index)),
                max_dif_S4=float(dif.max()) if len(dif) else None, minutos_con_dif_gt_1e9=int((dif > 1e-9).sum()))
    for f in ("Fix0", "Fix1", "Fix3", "Fix4"):
        if f in r.columns:
            fila[f"{f}_reproducido"] = int(r[f].astype(bool).sum())
            fila[f"{f}_archivo"] = int(ref[f].astype(str).str.lower().isin(["true", "1"]).sum()) if f in ref.columns else None
    filas.append(fila)
T = pd.DataFrame(filas)
out = [guardar(T, "V01b_reproduccion_reconstruccion.csv", "Reproducción de la reconstrucción frente al archivo usado")]
igual = T[(T.solo_reproducidas == 0) & (T.solo_archivo == 0) & (T.minutos_con_dif_gt_1e9 == 0)]
print("\nCONCLUSIÓN: " + (f"el archivo se reproduce EXACTAMENTE con max_gap = {int(igual.max_gap.iloc[0])}" if len(igual)
                         else "ninguna configuración reproduce exactamente el archivo; revisar las diferencias"))
bitacora("V01b", a.config, out)
