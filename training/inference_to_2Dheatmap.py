import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.layers import Average
from tensorflow.keras.models import Model
import sys

import os
import functions as fn

from tensorflow.keras.preprocessing.image import ImageDataGenerator

### XAI section ###

import innvestigate
import pickle as pkl
import numpy as np
import time

import tensorflow.keras.backend as K
import gc

import argparse



tf.compat.v1.disable_eager_execution() # for use iNNvestigate

# Argumentos del script
parser = argparse.ArgumentParser(description="Generador de mapas XAI por lotes")
parser.add_argument('--lote', type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help='Número de lote a procesar (1-5)')
args = parser.parse_args()

COLORMODE = "rgb"
RESNET = True

IMG_SHAPE = (36,108)
depth = 3
if RESNET:
    cnn = fn.create_CNN_Resnet
    model_weights = "entrenado/221512/model_rgb_resnet/model_rgb_resnet"
else:
    cnn = fn.create_CNN
    model_weights = "/content/drive/MyDrive/XAI_forense/training_models/221829/model_rgb_panoramacnn/model_rgb_panoramacnn"

print(model_weights)

### full model ###

cnn_0 = cnn(0,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_1 = cnn(1,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_2 = cnn(2,IMG_SHAPE[0],IMG_SHAPE[1],depth)

full_y = Average()([cnn_0.output, cnn_1.output, cnn_2.output])
model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=full_y)
model.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

model.load_weights(model_weights)
print("...loaded weights")

### single models ###

# ResNet
# x-axis
model_axis_x = cnn("x",IMG_SHAPE[0],IMG_SHAPE[1],depth)
model_axis_x.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

for layer_index, layer in enumerate(model_axis_x.layers):
    layer_index_model = 3*layer_index+3
    model_axis_x.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
    print('Layer:',layer_index, layer.name)
    print('Layer:',layer_index_model, model.layers[layer_index_model].name)

# y-axis
model_axis_y = cnn("y",IMG_SHAPE[0],IMG_SHAPE[1],depth)
model_axis_y.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

for layer_index, layer in enumerate(model_axis_y.layers):
    layer_index_model = 3*layer_index+4
    model_axis_y.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
    # print('Layer:',layer_index, layer.name)
    # print('Layer:',layer_index_model, model.layers[layer_index_model].name)

# z-axis
model_axis_z = cnn("z",IMG_SHAPE[0],IMG_SHAPE[1],depth)
model_axis_z.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

for layer_index, layer in enumerate(model_axis_z.layers):
    layer_index_model = 3*layer_index+5
    model_axis_z.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
    # print('Layer:',layer_index, layer.name)
    # print('Layer:',layer_index_model, model.layers[layer_index_model].name)

# #Panorama-CNN
# #x-axis
# model_axis_x = cnn("x",IMG_SHAPE[0],IMG_SHAPE[1],depth)
# model_axis_x.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

# for layer_index, layer in enumerate(model_axis_x.layers):
#     layer_index_model = 3*layer_index #+3
#     model_axis_x.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
#     print('Layer:',layer_index, layer.name)
#     print('Layer:',layer_index_model, model.layers[layer_index_model].name)

# #y-axis
# model_axis_y = cnn("y",IMG_SHAPE[0],IMG_SHAPE[1],depth)
# model_axis_y.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

# for layer_index, layer in enumerate(model_axis_y.layers):
#     layer_index_model = 3*layer_index +1#+4
#     model_axis_y.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
#     print('Layer:',layer_index, layer.name)
#     print('Layer:',layer_index_model, model.layers[layer_index_model].name)

# #z-axis
# model_axis_z = cnn("z",IMG_SHAPE[0],IMG_SHAPE[1],depth)
# model_axis_z.compile(loss=MeanAbsoluteError(), optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=['mae'])

# for layer_index, layer in enumerate(model_axis_z.layers):
#     layer_index_model = 3*layer_index +2#+5
#     model_axis_z.layers[layer_index].set_weights(model.layers[layer_index_model].get_weights())
#     print('Layer:',layer_index, layer.name)
#     print('Layer:',layer_index_model, model.layers[layer_index_model].name)




### XAI section ###

mth = 'integrated_gradients' # 'gradient' (panorama), 'integrated_gradients' (resnet)


def ig_xai(analyzer, dg, outfile):
    print(f"[XAI] Preparando entrada para: {outfile}", flush=True)
    images = next(dg)
    print(f"[XAI] Iniciando análisis de: {outfile}", flush=True)
    start = time.time()
    try:
        result = analyzer.analyze(images)
        duration = time.time() - start
        print(f"[XAI] Finalizado en {duration:.2f} segundos: {outfile}", flush=True)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as f:
            pkl.dump(result, f)
    except Exception as e:
        print(f"[ERROR] Falló el análisis de {outfile}: {e}", flush=True)

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

individuos_name = lotes[args.lote - 1]  # elige el lote correspondiente
print(f"Procesando lote {args.lote}: {[i[0] for i in individuos_name]}")

model = "resnet"# "panorama"

folder = "/mnt/homeGPU/jarodriguez/panorama_extended/pubis_data_proc_25_panorama_norm"
IMG_SHAPE2 = (108,36)

for indv in range(len(individuos_name)):
    name = individuos_name[indv][0]
    age = individuos_name[indv][1]
    ruta = "inferencia_img/" + model + "/2d_heatmaps/" + name + "/"

    test_x = [[name + "/" + name + "_0" + '_panorama_ext_X.png', age]]
    test_y = [[name + "/" + name + "_0" + '_panorama_ext_Y.png', age]]
    test_z = [[name + "/" + name + "_0" + '_panorama_ext_Z.png', age]]

    # Asegura que la carpeta existe
    os.makedirs(ruta, exist_ok=True)

    # === X-axis ===
    datagen = ImageDataGenerator()
    datagen_x_for_pred = fn.image_generator2(test_x, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
    datagen_x_for_xai = fn.image_generator2(test_x, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)

    est_x = model_axis_x.predict(datagen_x_for_pred, verbose=2, steps=1)
    print("Age estimation by X: {}".format(est_x[0][0]))

    # analyzer = innvestigate.create_analyzer(mth, model_axis_x)
    # ig_xai(analyzer, datagen_x_for_xai, ruta + name + '_X.pkl')
    output_file = ruta + name + '_X.pkl'
    if os.path.exists(output_file):
        print(f"[INFO] Ya existe {output_file}, se omite.")
    else:
        analyzer = innvestigate.create_analyzer(mth, model_axis_x)
        ig_xai(analyzer, datagen_x_for_xai, output_file)

    # === Y-axis ===
    datagen = ImageDataGenerator()
    datagen_y_for_pred = fn.image_generator2(test_y, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
    datagen_y_for_xai = fn.image_generator2(test_y, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)

    est_y = model_axis_y.predict(datagen_y_for_pred, verbose=2, steps=1)
    print("Age estimation by Y: {}".format(est_y[0][0]))

    # analyzer = innvestigate.create_analyzer(mth, model_axis_y)
    # ig_xai(analyzer, datagen_y_for_xai, ruta + name + '_Y.pkl')

    output_file = ruta + name + '_Y.pkl'
    if os.path.exists(output_file):
        print(f"[INFO] Ya existe {output_file}, se omite.")
    else:
        analyzer = innvestigate.create_analyzer(mth, model_axis_y)
        ig_xai(analyzer, datagen_y_for_xai, output_file)

    # === Z-axis ===
    datagen = ImageDataGenerator()
    datagen_z_for_pred = fn.image_generator2(test_z, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)
    datagen_z_for_xai = fn.image_generator2(test_z, folder, 1, datagen, IMG_SHAPE2, colormode=COLORMODE, shuffle=False, weights=False)

    est_z = model_axis_z.predict(datagen_z_for_pred, verbose=2, steps=1)
    print("Age estimation by Z: {}".format(est_z[0][0]))

    # analyzer = innvestigate.create_analyzer(mth, model_axis_z)
    # ig_xai(analyzer, datagen_z_for_xai, ruta + name + '_Z.pkl')

    output_file = ruta + name + '_Z.pkl'
    if os.path.exists(output_file):
        print(f"[INFO] Ya existe {output_file}, se omite.")
    else:
        analyzer = innvestigate.create_analyzer(mth, model_axis_z)
        ig_xai(analyzer, datagen_z_for_xai, output_file)




model = "resnet"# "panorama"
print("\n--- Postprocesado: generación de mapas 2D normalizados ---\n")

def normalized(nparray):
    nparray /= np.max(np.abs(nparray))  # normaliza [-1, 1]
    nparray[nparray < 0] = 0            # [0,1]
    return nparray

for name, _ in individuos_name:
    axes = ["X", "Y", "Z"]
    for axis in axes:
        indv_name = f"{name}_{axis}"
        pkl_path = f"inferencia_img/{model}/2d_heatmaps/{name}/{indv_name}.pkl"
        pkl_out = f"inferencia_img/{model}/2d_heatmaps/{name}/{indv_name}_2channels.pkl"

        if not os.path.exists(pkl_path):
            print(f"[AVISO] Archivo no encontrado, se omite: {pkl_path}")
            continue

        with open(pkl_path, 'rb') as f:
            hmap_3_channels = pkl.load(f)

        hmap_2_channels = hmap_3_channels.sum(axis=np.argmax(np.asarray(hmap_3_channels.shape) == 3))
        hmap_2_channels = normalized(hmap_2_channels)
        hmap_2_channels = np.transpose(hmap_2_channels[0])  # (36, 108)

        with open(pkl_out, 'wb') as f:
            pkl.dump(hmap_2_channels, f)

        print(f"[OK] Guardado: {pkl_out}")


hmaps_2_channels = []

for name, _ in individuos_name:
    for axis in ["X", "Y", "Z"]:
        pkl_file = f"inferencia_img/{model}/2d_heatmaps/{name}/{name}_{axis}_2channels.pkl"
        if not os.path.exists(pkl_file):
            print(f"[AVISO] Fichero no encontrado: {pkl_file}")
            continue
        with open(pkl_file, 'rb') as f:
            hmap_2_channels = pkl.load(f)  # (108, 36)
            print(hmap_2_channels.shape)
            hmap_2_channels = np.transpose(hmap_2_channels)  # (36, 108)
            print(hmap_2_channels.shape)
            hmaps_2_channels.append(hmap_2_channels)
