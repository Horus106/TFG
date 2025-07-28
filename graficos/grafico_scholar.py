import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("scholar.csv")
df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

# Función para recortar cada serie desde su primer valor > 0
def recortar_serie(df, columna):
    sub_df = df.copy()
    for i in range(len(sub_df)):
        if sub_df.loc[i, columna] > 0:
            return sub_df.iloc[i:][["anio", columna]]
    return df[["anio", columna]]  # por si todos son ceros

# Recortar cada una
base = recortar_serie(df, "basico")
ia = recortar_serie(df, "ia_base")
xai = recortar_serie(df, "xai")

# Graficar
plt.figure(figsize=(10, 6))

plt.plot(base["anio"], base["basico"], label="Total publicaciones", marker='o', color="dodgerblue")
plt.plot(ia["anio"], ia["ia_base"], label="Con IA", marker='o', color="darkorange")
plt.plot(xai["anio"], xai["xai"], label="Con XAI", marker='o', color="crimson")

plt.xlabel("Año")
plt.ylabel("Número de publicaciones")
plt.title("Tendencia de publicaciones sobre estimación de edad / perfil biológico")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("grafico_scholar_correcto.png", dpi=300)
print("✅ Gráfico guardado como 'grafico_scholar_correcto.png'")
plt.show()
