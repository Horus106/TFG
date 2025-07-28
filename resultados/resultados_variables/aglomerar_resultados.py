import pandas as pd
import os

# === Carpeta donde están los CSVs ===
carpeta = "./"  # cámbiala si están en otra ruta

# === Lista para guardar los resultados
resultado = []

# === Leer todos los CSVs de resultados
for archivo in os.listdir(carpeta):
    if archivo.startswith("execution_results_RESNET50_") and archivo.endswith(".csv"):
        ruta = os.path.join(carpeta, archivo)

        # Extraer identificador del modelo desde el nombre del archivo
        nombre_modelo = archivo.replace("execution_results_RESNET50_", "").replace(".csv", "")

        # Leer archivo CSV
        df = pd.read_csv(ruta, sep=";")

        # Insertar columna 'modelo' al principio
        df.insert(0, "modelo", nombre_modelo)

        df.drop(df.columns[1], axis=1, inplace=True)


        # Añadir al conjunto total
        resultado.append(df)

# === Concatenar todo
df_final = pd.concat(resultado, ignore_index=True)

# === Guardar CSV combinado
df_final.to_csv("resultados_combinados.csv", sep=";", index=False)
print("✅ Archivo generado: resultados_combinados.csv")
