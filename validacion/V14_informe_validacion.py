"""V14 · Informe consolidado: une las salidas de V00–V13 en validacion/salidas/INFORME_VALIDACION.md,
con la correspondencia observación de auditoría → script → archivo de evidencia.
Uso: python validacion/V14_informe_validacion.py"""
from comun import *
MAPA = [("Reproducibilidad (commit, huellas)", "V00", ["V00_entorno.csv", "V00_huellas_archivos.csv"]),
        ("C1-017 / C1-018 Reconstrucción y días descartados", "V01", ["V01_resumen_serie.csv", "V01_etapas_fix.csv", "V01_conciliacion_dias_lt50.csv", "V01_reconstruidos_franja_2346.csv"]),
        ("C1-020 / C1-021 / Resumen 1 % Particiones, eventos, desbalance", "V02", ["V02_particiones.csv", "V02_eventos_por_definicion.csv", "V02_desbalance_ventanas.csv"]),
        ("C1-016 Reconstruidos en la entrada X", "V03", ["V03_reconstruidos_en_X.csv"]),
        ("C1-019 Retención de precursores", "V04", ["V6_relleno_mas_alla_15min.csv", "V6_minutos_sin_precursor.csv", "V6_ventanas_por_particion.csv"]),
        ("C1-023 Escala de θ y ρ; exponente de la WFL", "V05", ["V05_escalador_theta_rho.csv", "V05_pesos_wfl.csv"]),
        ("C1-014 Elevación del satélite del máximo", "V06", ["V06_elevacion_en_eventos.csv", "V06_resumen_elevacion.csv", "V06_cambios_satelite.csv"]),
        ("C1-031 Autocorrelación nocturna y sin ciclo", "V07", ["V07_primer_lag_acf_lt05.csv"]),
        ("C1-016/022/027/028/029/030 Re-análisis en prueba", "V08", ["V08_metricas_prueba.csv", "V08_hss_ic95.csv", "V08_inicios_por_evento.csv", "V08_prospectivo_22jul_30sep.csv"]),
        ("C1-026 Semillas del v3.0 en prueba", "V09", ["V09_semillas_v30_resumen.csv"]),
        ("C1-025 β con la configuración final", "V11", ["V11_beta_final.csv"]),
        ("C1-024 / C1-011 Ablación de la WFL y sesgo", "V12", ["V12_ablacion_wfl.csv"]),
        ("C1-019 Experimentos con precursores corregidos (antes y v6)", "V13", ["V13_a_precursores_vs_C.csv", "V13_b_importancia_ROTI.csv", "V13_c_meseta_beta.csv", "V13_d_lstm_patchtst.csv"])]
L = [f"# Informe de validación de la tesis\n\nGenerado: {pd.Timestamp.now():%Y-%m-%d %H:%M} · commit `{commit()}`\n",
     "| Observación | Script | Evidencia | Estado |", "|---|---|---|---|"]
for obs, sc, fs in MAPA:
    ok = [f for f in fs if (SAL / f).exists()]
    L.append(f"| {obs} | {sc} | {', '.join(ok) or '—'} | {'generado' if len(ok) == len(fs) else ('parcial' if ok else 'pendiente')} |")
for obs, sc, fs in MAPA:
    L.append(f"\n## {sc} · {obs}\n")
    for f in fs:
        p = SAL / f
        if p.exists():
            df = pd.read_csv(p)
            try:
                tabla = df.head(40).to_markdown(index=False)
            except ImportError:
                tabla = "```\n" + df.head(40).to_string(index=False) + "\n```"
            L.append(f"**{f}**\n\n{tabla}\n")
(SAL / "INFORME_VALIDACION.md").write_text("\n".join(L), encoding="utf8")
print(f"Informe: {SAL / 'INFORME_VALIDACION.md'}")
