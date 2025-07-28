import os
import pandas as pd
import matplotlib.pyplot as plt

# Configurar estilo profesional
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# Carpeta donde están los históricos
carpeta = "./"

print(f"\n📁 Carpeta: {os.path.abspath(carpeta)}\n")

for archivo in os.listdir(carpeta):
    if archivo.endswith(".csv"):
        ruta_csv = os.path.join(carpeta, archivo)
        nombre_base = os.path.splitext(archivo)[0]
        print(f"\n🔍 Procesando archivo: {archivo}")

        try:
            # Leer el archivo
            df = pd.read_csv(ruta_csv, sep=";")
            primera_col = df.columns[0]
            if "Unnamed" in primera_col or primera_col == "":
                df.rename(columns={primera_col: "epoch"}, inplace=True)

            df.set_index("epoch", inplace=True)

            # Detectar métricas
            loss_cols = [col for col in ["loss", "val_loss"] if col in df.columns]
            mae_cols = [col for col in ["mae", "val_mae"] if col in df.columns]

            if not loss_cols and not mae_cols:
                print("⚠️ No hay métricas conocidas. Saltando.")
                continue

            # Crear figura
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            if loss_cols:
                for col in loss_cols:
                    axs[0].plot(df.index, df[col], label=col)
                axs[0].set_ylabel("Loss")
                axs[0].legend()
                axs[0].set_title(f"Histórico de entrenamiento: {nombre_base}")

            if mae_cols:
                for col in mae_cols:
                    axs[1].plot(df.index, df[col], label=col)
                axs[1].set_ylabel("MAE")
                axs[1].legend()

            axs[1].set_xlabel("Épocas")
            plt.tight_layout()

            salida = os.path.join(carpeta, f"{nombre_base}.png")
            plt.savefig(salida, dpi=300)
            plt.close()
            print(f"✅ Imagen guardada: {salida}")

        except Exception as e:
            print(f"❌ ERROR procesando {archivo}: {e}")
