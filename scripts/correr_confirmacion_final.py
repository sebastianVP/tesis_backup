"""
PUNTO DE CONTROL · Confirmación de la configuración final en VALIDACIÓN.

Entrena S4 + hora con la pérdida final (WFL α = 1.0, β = 5, elegida por la regla
pre-registrada en E4) y ventanas de 30 y 60 min (decisión pendiente de E2),
con 3 semillas y umbral de alerta calibrado. No usa la partición de prueba.

Regla para la ventana: la más corta cuyo HSS calibrado esté a menos de una
desviación estándar del mejor.

Uso (desde la raíz del proyecto, cuando la GPU esté libre):
    nohup python scripts/correr_confirmacion_final.py > experiments/log_control_ventana.txt 2>&1 &
"""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from s4.config import cargar_config, variante, ruta
from s4.pipeline import preparar_datos, correr_variante
from s4.registry import leer, resumir_semillas
from s4.train import configurar_gpu
from s4 import features as ft

configurar_gpu()
base = variante(cargar_config("config/tesis_final.yaml"), **{"perdida.alpha": 1.0, "perdida.beta": 5.0})
print("Configuración: S4 + hora · WFL α = 1.0, β = 5 · umbral calibrado · 3 semillas", flush=True)
datos = preparar_datos(base, usar_precursores=False)        # igual que el modelo final: solo S4 y la hora

for lb in [30, 60]:
    print(f"\n===== C · S4 + hora · LB{lb} · a1_b5 =====", flush=True)
    correr_variante(variante(base, **{"ventanas.lookback": lb}), datos, ft.CONJUNTOS["B_S4_ciclica"],
                    f"C · LB{lb} · a1_b5", "CONTROL_ventana", barrido_umbral=True)

reg = leer(ruta(base, "experimentos") / f"registro_{base['nombre']}.csv", grupo="CONTROL_ventana", particion="val")
t = resumir_semillas(reg, por=("modelo",),
                     metricas=("HSS_opt", "umbral_opt", "Precision_opt", "POD_inicio_opt", "RMSE_evento", "RMSE_global"))
t.to_csv(ruta(base, "reportes") / "tables/CONTROL_ventana.csv", index=False)
print("\n=== CONTROL_ventana ===\n" + t.round(3).to_string(index=False))
print("\nPUNTO DE CONTROL TERMINADO", flush=True)