"""V10a · Parche v6.0: retención de 15 min para TEC, ROTEC y ROTI dentro del paquete.

Problema: features.causalizar_tramos_lineales() termina en out.ffill() SIN límite, y
se aplica sobre TEC/ROTEC/ROTI que el integrador (o io.cargar_glotec) YA dejó causales
con un máximo de 15 min. Cambiarlo por ffill(limit=15) NO basta: el último minuto
mantenido por el integrador cuenta como dato y se suman otros 15 min (30 en total).

Solución: nueva opción datos.recausalizar_locales (por defecto True = comportamiento
anterior, para reproducir los registros previos). Con False, TEC/ROTEC/ROTI se usan tal
como vienen del integrador v3 o de cargar_glotec(limite_min=15): retención de 15 min.

Uso (desde la raíz del proyecto):  python validacion/V10a_aplicar_parche_retencion.py
"""
from pathlib import Path

f = Path("src/s4/features.py"); s = f.read_text(encoding="utf8")
if "recausalizar_locales" not in s:
    s = s.replace("causal: bool = True, verbose: bool = True, ffill_max_min: int = 60) -> pd.DataFrame:",
                  "causal: bool = True, verbose: bool = True, ffill_max_min: int = 60,\n"
                  "                      recausalizar_locales: bool = True) -> pd.DataFrame:", 1)
    viejo = "            for col in LOCALES:\n                if col in mv.columns:"
    assert viejo in s, "No se encontró el bucle de LOCALES en construir_dataset()"
    s = s.replace(viejo, "            for col in (LOCALES if recausalizar_locales else []):\n                if col in mv.columns:", 1)
    f.write_text(s, encoding="utf8"); print("features.py: parcheado")
else:
    print("features.py: ya tenía el parche")

f = Path("src/s4/pipeline.py"); s = f.read_text(encoding="utf8")
if "recausalizar_locales" not in s:
    viejo = "ffill_max_min=d.get(\"ffill_precursores_min\", 60))"
    assert viejo in s, "No se encontró la llamada a construir_dataset() en preparar_datos()"
    s = s.replace(viejo, "ffill_max_min=d.get(\"ffill_precursores_min\", 60),\n"
                         "                              recausalizar_locales=d.get(\"recausalizar_locales\", True))", 1)
    f.write_text(s, encoding="utf8"); print("pipeline.py: parcheado")
else:
    print("pipeline.py: ya tenía el parche")

cfg = Path("config/tesis_final.yaml").read_text(encoding="utf8")
nuevo = Path("config/tesis_v6.yaml")
if not nuevo.exists():
    import re
    cfg = re.sub(r"(?m)^nombre:.*$", "nombre: tesis_v6", cfg)
    cfg = re.sub(r"(?m)^(\s*)ffill_precursores_min:.*$", r"\1ffill_precursores_min: 0\n\1recausalizar_locales: false", cfg)
    assert "recausalizar_locales: false" in cfg and "nombre: tesis_v6" in cfg
    nuevo.write_text(cfg, encoding="utf8"); print("config/tesis_v6.yaml: creado (registro_tesis_v6.csv)")
else:
    print("config/tesis_v6.yaml: ya existe")
