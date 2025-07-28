import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import innvestigate
import pickle as pkl
import numpy as np
import os
import argparse
import time
import functions as fn

tf.compat.v1.disable_eager_execution()  # Necesario para iNNvestigate

# === Argumentos ===
parser = argparse.ArgumentParser(description="Generador de mapas XAI por lote y vista")
parser.add_argument('--lote', type=int, required=True, choices=[1, 2, 3, 4, 5], help='Número de lote a procesar (1-5)')
parser.add_argument('--view', type=str, required=True, choices=['X', 'Y', 'Z'], help='Vista a analizar: X, Y o Z')
parser.add_argument('--model_weights', type=str, required=True, help='Ruta al archivo de pesos del modelo entrenado')
args = parser.parse_args()

# === Configuración ===
VIEW = args.view.upper()
COLORMODE = "rgb"
IMG_SHAPE = (36, 108)
IMG_SHAPE_MODEL = (108, 36)  # como se cargan las imágenes
DEPTH = 3
cnn = fn.create_CNN_Resnet  # cambiar si no usas ResNet

# === Modelo individual por vista ===
model = cnn(0, IMG_SHAPE[0], IMG_SHAPE[1], DEPTH)
model.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(1e-3), metrics=['mae'])
model.load_weights(args.model_weights)
print(f"[INFO] Modelo cargado para vista {VIEW} desde: {args.model_weights}")

# === Configuración XAI ===
mth = 'integrated_gradients'  # o 'gradient'
folder = "/mnt/homeGPU/jarodriguez/panorama_extended/pubis_data_proc_25_panorama_norm"
model_name = os.path.basename(args.model_weights).split('.')[0]

# resnet
# individuos_id = [6,14,15,48,64,141,149,177,14,64,177,271,303,313,367,380,504]
# individuos_name = [("6_Izq",19),("14_Dch",33),("15_Izq",23),("48_Dch",30),("64_Dch",52),
#                   ("141_Izq",20),("149_Izq",50),("177_Izq",16),("14_Dch",33),("64_Dch",52),("177_Izq",16),("271_Izq",47),("303_Dch",49),("313_Izq",30),("367_Izq",18),("380_Dch",69),("504_Izq",35)]

individuos_id = [225, 558]
individuos_name = [("225_Dch",41),("558_Izq",26)]


# # panorama
# individuos_id = [243,505,141,313,341,66,325,303,403,380,380,272,6,186,530,231]
# individuos_name = [("243_Izq",17),("505_Dch",18),("141_Izq",20),("313_Izq",30),("341_Izq",29),("66_Izq",34),
#                    ("325_Dch",47),("303_Dch",49),("403_Dch",52),("380_Dch",69),("380_Izq",69),("272_Izq",82),
#                    ("6_Izq",19),("186_Dch",37),("530_Dch",44),("231_Izq",56)]

# División en lotes: 4 + 4 + 4 + 4 + 5
lotes = [
    individuos_name[0:2],
    individuos_name[4:8],
    individuos_name[8:12],
    individuos_name[12:16],
    individuos_name[16:]  # el último lote con 1
]
individuos_name = lotes[args.lote - 1]
print(f"[INFO] Lote {args.lote} seleccionado. Vista: {VIEW}")

# === Función XAI ===
def ig_xai(analyzer, dg, outfile):
    print(f"[XAI] Generando: {outfile}")
    images = next(dg)
    try:
        result = analyzer.analyze(images)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pkl.dump(result, f)
        print(f"[OK] Guardado: {outfile}")
    except Exception as e:
        print(f"[ERROR] Fallo en {outfile}: {e}")

# === Bucle por individuo ===
for name, age in individuos_name:
    ruta = f"inferencia_img/{model_name}/2d_heatmaps/{name}/"
    img_path = f"{name}/{name}_0_panorama_ext_{VIEW}.png"
    test_data = [[img_path, age]]

    output_file = os.path.join(ruta, f"{name}_{VIEW}.pkl")
    if os.path.exists(output_file):
        print(f"[INFO] Ya existe {output_file}, se omite.")
        continue

    datagen = ImageDataGenerator()
    datagen_pred = fn.image_generator2(test_data, folder, 1, datagen, IMG_SHAPE_MODEL, colormode=COLORMODE, shuffle=False, weights=False)
    datagen_xai = fn.image_generator2(test_data, folder, 1, datagen, IMG_SHAPE_MODEL, colormode=COLORMODE, shuffle=False, weights=False)

    est = model.predict(datagen_pred, verbose=0, steps=1)
    print(f"[INFO] Estimación de edad para {name}: {est[0][0]}")

    analyzer = innvestigate.create_analyzer(mth, model)
    ig_xai(analyzer, datagen_xai, output_file)

# === Normalización del heatmap ===
def normalized(nparray):
    nparray /= np.max(np.abs(nparray))
    nparray[nparray < 0] = 0
    return nparray

for name, _ in individuos_name:
    pkl_path = f"inferencia_img/{model_name}/2d_heatmaps/{name}/{name}_{VIEW}.pkl"
    pkl_out = f"inferencia_img/{model_name}/2d_heatmaps/{name}/{name}_{VIEW}_2channels.pkl"

    if not os.path.exists(pkl_path):
        print(f"[AVISO] No se encuentra: {pkl_path}")
        continue

    with open(pkl_path, 'rb') as f:
        hmap_3 = pkl.load(f)

    hmap_2 = hmap_3.sum(axis=np.argmax(np.asarray(hmap_3.shape) == 3))
    hmap_2 = normalized(hmap_2)
    hmap_2 = np.transpose(hmap_2[0])

    with open(pkl_out, 'wb') as f:
        pkl.dump(hmap_2, f)
    print(f"[OK] Heatmap 2D guardado: {pkl_out}")
