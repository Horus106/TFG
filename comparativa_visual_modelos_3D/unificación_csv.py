import pandas as pd
import os

# Pedimos al usuario que introduzca la ruta
ruta_csvs = input("Introduce la ruta donde están los CSVs: ").strip()

# Lista para almacenar los DataFrames
dataframes = []

# Recorremos todos los archivos CSV de la carpeta
for nombre_archivo in os.listdir(ruta_csvs):
    if nombre_archivo.endswith(".csv"):
        ruta_completa = os.path.join(ruta_csvs, nombre_archivo)
        modelo = os.path.splitext(nombre_archivo)[0]  # nombre sin extensión
        df = pd.read_csv(ruta_completa)
        df["modelo"] = modelo
        dataframes.append(df)

# Unimos todos los DataFrames en uno solo
df_unido = pd.concat(dataframes, ignore_index=True)

# Guardamos el resultado en un nuevo archivo CSV
output_path = os.path.join(ruta_csvs, "resultado_unificado.csv")
df_unido.to_csv(output_path, index=False)

print(f"CSV unificado creado correctamente en: {output_path}")
