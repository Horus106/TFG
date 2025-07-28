import requests
import time
import pandas as pd
import os

CSV_FILE = "publicaciones_estimacion_edad.csv"

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HEADERS = {
    "User-Agent": "TFG Estudiante UGR"
    # Si tienes una API key, añade aquí: "x-api-key": "TU_API_KEY"
}

consulta = '("age estimation" OR "age-at-death estimation" OR "biological profile" OR "biological profile estimation")'

# Cargar CSV si existe
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    df = df.drop_duplicates(subset="año")
else:
    df = pd.DataFrame(columns=["año", "num_publicaciones"])

# Años objetivo
años_completos = list(range(1975, 2026))

# Años ya con datos válidos
años_existentes = df[df["num_publicaciones"].notna()]["año"].tolist()

# Años que faltan por consultar
años_faltantes = [a for a in años_completos if a not in años_existentes]

print(f"Años pendientes de consulta: {años_faltantes}")

for i, año in enumerate(años_faltantes):
    query_text = f'{consulta} year:{año}'
    params = {
        "query": query_text,
        "limit": 1,
        "offset": 0,
        "fields": "year"
    }

    try:
        response = requests.get(API_URL, params=params, headers=HEADERS)
        if response.status_code == 200:
            total = response.json().get("total", 0)
            print(f"Año {año}: {total} publicaciones")
            df = pd.concat([df, pd.DataFrame([{"año": año, "num_publicaciones": total}])], ignore_index=True)
        else:
            print(f"Error en el año {año}: {response.status_code}")
            df = pd.concat([df, pd.DataFrame([{"año": año, "num_publicaciones": None}])], ignore_index=True)
    except Exception as e:
        print(f"Excepción en el año {año}: {e}")
        df = pd.concat([df, pd.DataFrame([{"año": año, "num_publicaciones": None}])], ignore_index=True)

    # Pausa cada 5 peticiones para evitar 429
    if (i + 1) % 5 == 0:
        print("Esperando 60 segundos para evitar límite de peticiones...")
        time.sleep(60)

# Guardar sin duplicados
df = df.drop_duplicates(subset="año")
df = df.sort_values("año")
df.to_csv(CSV_FILE, index=False)
print("✅ CSV actualizado:", CSV_FILE)
