import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# === Carpeta de trabajo
carpeta = "./"

# === Diccionario para guardar los datos
mae_por_modelo = {}

# === Buscar archivos válidos
for archivo in os.listdir(carpeta):
    if archivo.startswith("history_ft_") and archivo.endswith(".csv"):
        ruta = os.path.join(carpeta, archivo)
        nombre_modelo = archivo.replace("history_ft_", "").replace(".csv", "")
        print(f"📄 Procesando {archivo} como modelo '{nombre_modelo}'")

        try:
            df = pd.read_csv(ruta, sep=";")
            # Renombrar columna sin nombre si aplica
            primera_col = df.columns[0]
            if "Unnamed" in primera_col or primera_col == "":
                df.rename(columns={primera_col: "epoch"}, inplace=True)
            df.set_index("epoch", inplace=True)

            # Preferir val_mae, si no existe usar mae
            if "val_mae" in df.columns:
                serie = df["val_mae"]
            elif "mae" in df.columns:
                serie = df["mae"]
            else:
                print(f"⚠️ {archivo} no tiene 'mae' ni 'val_mae'. Saltado.")
                continue

            mae_por_modelo[nombre_modelo] = serie

        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")

# === Plotear la comparativa
if not mae_por_modelo:
    print("❌ No se encontró ninguna métrica MAE válida.")
    exit()

plt.figure(figsize=(12, 7))

for modelo, serie in mae_por_modelo.items():
    plt.plot(serie.index, serie.values, label=modelo)

plt.title("Comparativa de MAE entre modelos")
plt.xlabel("Épocas")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()

# === Guardar imagen
salida = os.path.join(carpeta, "comparativa_mae.png")
plt.savefig(salida, dpi=300)
plt.close()
print(f"✅ Comparativa guardada como: {salida}")
