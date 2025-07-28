import enum
from random import random
from re import S
from xml.etree.ElementInclude import include
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import callbacks, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, ZeroPadding2D, concatenate, Average
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16

from sklearn.model_selection import KFold

from keras.utils.vis_utils import plot_model

from collections import Counter

import scipy.ndimage as sci

import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2 as cv

import random

import ranges_of_age as roa

import sys

import math

from datetime import datetime

def create_CNN(n, width, height, depth, summary = False):
    input_shape = (height, width, depth)

    inputs = Input(shape=input_shape, name=f'input_{n}')

    filters = [64, 256, 1024]
    f_size = [5 ,5, 3]

    x = inputs

    for i, f in enumerate(filters):
        x = ZeroPadding2D(padding=2, name=f'zero_pad_{n}_{i}')(x)
        x = Conv2D(f, (f_size[i],f_size[i]), padding="valid", activation='ReLU', name=f'conv_{n}_{i}')(x)
        x = BatchNormalization(name=f'batch_norm_{n}_{i}')(x)
        x = MaxPooling2D(pool_size=(2,2), name=f'max_pool_{n}_{i}')(x)

    x = Flatten(name=f'flatten_{n}')(x)
    x = Dense(100, activation='ReLU', name=f'dense1_{n}')(x)
    x = Dense(100, activation='ReLU', name=f'dense2_{n}')(x)
    x = Dropout(0.1, name=f'dropout_{n}')(x)
    x = Dense(1, activation='ReLU', name=f'output_{n}')(x)
    

    model = Model(inputs, x, name=f'panorama_cnn_{n}')

    if summary:
        model.summary()

    return model

# def create_CNN_Resnet(n, width, height, depth, trainable=False, summary=False):
#     input_shape = (height, width, depth)
#     inputs = Input(shape=input_shape, dtype='float32', name=f'input_{n}')


#     # Cargar modelo base
#     base_model = ResNet50(include_top=False, input_tensor=inputs, input_shape=input_shape)

#     # Renombrar todas las capas para evitar colisiones
#     for layer in base_model.layers:
#         layer._name = f"{layer.name}_{n}"
#     base_model._name = f"resnet50_{n}"
#     base_model.trainable = trainable

#     # Añadir capas propias
#     x = base_model.output
#     x = Flatten(name=f'flatten_{n}')(x)
#     x = Dense(100, activation='relu', name=f'dense1_{n}')(x)
#     x = Dense(100, activation='relu', name=f'dense2_{n}')(x)
#     x = Dropout(0.1, name=f'dropout_{n}')(x)
#     x = Dense(1, activation='relu', name=f'output_{n}')(x)
   

#     model = Model(inputs=inputs, outputs=x, name=f'resnet_model_{n}')

#     if summary:
#         model.summary()

#     return model

def create_CNN_Resnet(n, width, height, depth, trainable=False, summary=False):
    input_shape = (height, width, depth)
    inputs = Input(shape=input_shape, dtype='float32', name=f'input_{n}')

    # CRÍTICO: Crear modelo base sin input_tensor para evitar conflictos
    base_model = ResNet50(
        include_top=False, 
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Renombrar TODAS las capas del modelo base para evitar colisiones
    for i, layer in enumerate(base_model.layers):
        layer._name = f"{layer.name}_{n}_{i}"  # Añadir índice para garantizar unicidad
    
    base_model._name = f"resnet50_{n}"
    base_model.trainable = trainable

    # Conectar manualmente el input al modelo base
    x = base_model(inputs)
    
    # Añadir capas propias con nombres únicos
    x = Flatten(name=f'flatten_{n}')(x)
    x = Dense(100, activation='relu', name=f'dense1_{n}')(x)
    x = Dense(100, activation='relu', name=f'dense2_{n}')(x)
    x = Dropout(0.1, name=f'dropout_{n}')(x)
    x = Dense(1, activation='relu', name=f'output_{n}')(x)

    model = Model(inputs=inputs, outputs=x, name=f'resnet_model_{n}')

    if summary:
        model.summary()

    return model

def get_images_list(dataframe, colormode, projection_1, projection_2, augment=True, weights=True):
    """
    Adaptada para trabajar con solo 2 proyecciones específicas
    """
    list = dataframe.to_numpy()
    image_list = []

    # Crear sufijos para las dos proyecciones seleccionadas
    suffixes = []
    for projection in [projection_1, projection_2]:
        if colormode == 'rgb':
            suffix = f"_panorama_ext_{projection}.png"
        elif colormode == 'grayscale':
            suffix = {
                'X': '_panorama_SDM.png',
                'Y': '_panorama_NDM.png',
                'Z': '_panorama_GNDM.png'
            }[projection]
        suffixes.append(suffix)

    if augment:
        for t in list:
            for n in np.arange(20):
                imgs = [f"{t[1]}/{t[1]}_{n}{suffix}" for suffix in suffixes]
                imgs.append(t[2])  # Agregar label
                if weights:
                    imgs.append(t[3])  # Agregar weight si está disponible
                image_list.append(imgs)
    else:
        for t in list:
            imgs = [f"{t[1]}/{t[1]}_0{suffix}" for suffix in suffixes]
            imgs.append(t[2])  # Agregar label
            if weights:
                imgs.append(t[3])  # Agregar weight si está disponible
            image_list.append(imgs)

    return np.array(image_list)

def cached_img_read(img_path, img_shape, colormode, image_cache):
    """
    Función de lectura de imágenes con cache, corregida para evitar dtypes estructurados
    """
    if img_path not in image_cache:
        try:
            # Cargar imagen
            image = load_img(img_path, color_mode=colormode, target_size=img_shape, interpolation='bilinear')
            image_array = img_to_array(image)
            
            # Asegurar que sea numpy array limpio
            image_array = np.asarray(image_array, dtype=np.float32)
            
            # Normalizar a [0, 1]
            image_array = image_array / 255.0
            
            # Verificar que no tenga dtype estructurado
            if hasattr(image_array, 'dtype') and image_array.dtype.names is not None:
                print(f"[WARNING] Detectado dtype estructurado en {img_path}, limpiando...")
                if 'resource' in image_array.dtype.names:
                    image_array = image_array['resource']
                image_array = np.asarray(image_array.tolist(), dtype=np.float32)
            else:
                image_array = np.asarray(image_array, dtype=np.float32)

            # Conversión de color para RGB (si es necesario)
            if colormode == 'rgb' and image_array.shape[-1] == 3:
                image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
            
            # Asegurar forma correcta
            expected_channels = 1 if colormode == "grayscale" else 3

            if image_array.ndim == 2:
                # Imagen 2D → duplicar canales
                if expected_channels == 3:
                    image_array = np.stack([image_array] * 3, axis=-1)
                else:
                    image_array = np.expand_dims(image_array, axis=-1)

            elif image_array.ndim == 3:
                current_channels = image_array.shape[-1]
                if current_channels != expected_channels:
                    if expected_channels == 1:
                        image_array = np.mean(image_array, axis=-1, keepdims=True)
                    elif expected_channels == 3:
                        image_array = np.stack([image_array[..., 0]] * 3, axis=-1)
                    else:
                        print(f"[WARNING] Canales no esperados: {current_channels} en {img_path}")
            else:
                print(f"[ERROR] Dimensión inesperada en {img_path}: {image_array.shape}")
                image_array = np.zeros((img_shape[0], img_shape[1], expected_channels), dtype=np.float32)

            
            image_cache[img_path] = image_array
            
        except Exception as e:
            print(f"[ERROR] Error leyendo imagen {img_path}: {e}")
            # Crear imagen vacía en caso de error
            if colormode == "grayscale":
                image_array = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)
            else:
                image_array = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float32)
            image_cache[img_path] = image_array
    
    return image_cache[img_path]

def read_images_gen(images, dir, img_shape, datagen, colormode, image_cache, weights=True):
    """
    Adaptada para trabajar con número variable de imágenes de entrada (2 en este caso)
    """
    # Determinar cuántas imágenes hay por entrada (excluyendo label y weight)
    if weights:
        num_images = len(images[0]) - 2  # Excluir label y weight
    else:
        num_images = len(images[0]) - 1  # Excluir solo label

    # Crear arrays para cada imagen
    if colormode == "grayscale":
        X = [np.zeros((len(images), img_shape[0], img_shape[1], 1)) for _ in range(num_images)]
    else:
        X = [np.zeros((len(images), img_shape[0], img_shape[1], 3)) for _ in range(num_images)]

    for i, paths in enumerate(images):
        # Solo procesar las rutas de las imágenes (no el label ni el weight)
        img_paths = paths[:num_images]
        for v, img_path in enumerate(img_paths):
            full_path = os.path.join(dir, img_path)
            image = cached_img_read(full_path, img_shape, colormode, image_cache)
            image = datagen.standardize(image)
            X[v][i] = image

    return X

def image_generator_fixed(images, dir, batch_size, datagen, img_shape=(108,108), colormode="rgb", shuffle=True, weights=True):
    """
    Generador de imágenes corregido para evitar problemas de carga
    """
    image_cache = {}

    while True:
        n_imgs = len(images)
        if shuffle:
            indexs = np.random.permutation(np.arange(n_imgs))
        else:
            indexs = np.arange(n_imgs)
        
        num_batches = max(1, n_imgs // batch_size)

        for bid in range(num_batches):
            start_idx = bid * batch_size
            end_idx = min((bid + 1) * batch_size, n_imgs)
            
            batch = [images[i] for i in indexs[start_idx:end_idx]]
            actual_batch_size = len(batch)
            
            # Extraer información del batch
            if weights:
                img_paths = [b[:-2] for b in batch]  # Excluir label y weight
                labels = np.array([float(b[-2]) for b in batch], dtype=np.float32)
            else:
                img_paths = [b[:-1] for b in batch]  # Excluir solo label
                labels = np.array([float(b[-1]) for b in batch], dtype=np.float32)

            # Inicializar arrays para las dos proyecciones
            if colormode == "grayscale":
                X1 = np.zeros((actual_batch_size, img_shape[0], img_shape[1], 1), dtype=np.float32)
                X2 = np.zeros((actual_batch_size, img_shape[0], img_shape[1], 1), dtype=np.float32)
            else:
                X1 = np.zeros((actual_batch_size, img_shape[0], img_shape[1], 3), dtype=np.float32)
                X2 = np.zeros((actual_batch_size, img_shape[0], img_shape[1], 3), dtype=np.float32)

            # Cargar imágenes
            for i, paths in enumerate(img_paths):
                # Primera proyección
                full_path_1 = os.path.join(dir, paths[0])
                image_1 = cached_img_read(full_path_1, img_shape, colormode, image_cache)
                image_1 = datagen.standardize(image_1)
                X1[i] = image_1
                
                # Segunda proyección
                full_path_2 = os.path.join(dir, paths[1])
                image_2 = cached_img_read(full_path_2, img_shape, colormode, image_cache)
                image_2 = datagen.standardize(image_2)
                X2[i] = image_2

            yield ([X1, X2], labels)
        
def image_generator(images, dir, batch_size, datagen, img_shape=(108,108), colormode="rgb", shuffle=True, weights=True):
    """
    Generador de imágenes adaptado para trabajar con 2 proyecciones
    """
    image_cache = {}

    while True:
        n_imgs = len(images)
        indexs = np.random.permutation(np.arange(n_imgs)) if shuffle else np.arange(n_imgs)
        num_batches = n_imgs // batch_size

        for bid in range(num_batches):
            batch = [images[i] for i in indexs[bid*batch_size:(bid+1)*batch_size]]
            
            # Extraer rutas de imágenes y labels
            if weights:
                img_paths = [b[:-2] for b in batch]  # Excluir label y weight
                label = np.array([b[-2] for b in batch]).astype(np.float32)  # Label está en penúltima posición
            else:
                img_paths = [b[:-1] for b in batch]  # Excluir solo label
                label = np.array([b[-1] for b in batch]).astype(np.float32)  # Label está en última posición

            X = read_images_gen(img_paths, dir, img_shape, datagen, colormode, image_cache, weights)

            yield (X, label)

def image_generator2(images, dir, batch_size, datagen, img_shape=(108,108), colormode="rgb", shuffle=True, weights=True):
    """
    Generador optimizado para XAI - Maneja tanto modelos individuales como ensemble
    """
    image_cache = {}

    while True:
        n_imgs = len(images)
        indexs = np.random.permutation(np.arange(n_imgs)) if shuffle else np.arange(n_imgs)
        
        # Procesar todos los elementos disponibles
        for bid in range(max(1, (n_imgs + batch_size - 1) // batch_size)):  # Ceiling division
            # Calcular índices del batch
            start_idx = bid * batch_size
            end_idx = min((bid + 1) * batch_size, n_imgs)
            
            batch = [images[i] for i in indexs[start_idx:end_idx]]
            actual_batch_size = len(batch)
            
            # Extraer rutas de imágenes y labels
            if weights:
                img_paths = [b[:-2] for b in batch]  # Excluir label y weight
                label = np.array([b[-2] for b in batch], dtype=np.float32)
            else:
                img_paths = [b[:-1] for b in batch]  # Excluir solo label  
                label = np.array([b[-1] for b in batch], dtype=np.float32)

            # Determinar número de imágenes por entrada
            num_images = len(img_paths[0]) if img_paths else 1
            
            # Inicializar arrays para cada vista
            if colormode == "grayscale":
                X = [np.zeros((actual_batch_size, img_shape[0], img_shape[1], 1), dtype=np.float32) for _ in range(num_images)]
            else:
                X = [np.zeros((actual_batch_size, img_shape[0], img_shape[1], 3), dtype=np.float32) for _ in range(num_images)]

            # Cargar imágenes
            for i, paths in enumerate(img_paths):
                for v, img_path in enumerate(paths):
                    full_path = os.path.join(dir, img_path)
                    try:
                        image = cached_img_read(full_path, img_shape, colormode, image_cache)
                        image = datagen.standardize(image)
                        
                        # Asegurar tipo float32 limpio
                        image = np.asarray(image, dtype=np.float32)
                        
                        # Verificar dimensiones
                        expected_shape = (img_shape[0], img_shape[1], 1 if colormode == "grayscale" else 3)
                        if image.shape != expected_shape:
                            # Intentar ajustar dimensiones
                            if len(image.shape) == 2 and colormode == "grayscale":
                                image = np.expand_dims(image, axis=-1)
                            elif len(image.shape) == 3 and image.shape[2] == 1 and colormode == "rgb":
                                # Convertir grayscale a RGB duplicando canales
                                image = np.repeat(image, 3, axis=2)
                            
                            # Si aún no coincide, hacer reshape si es posible
                            if image.shape != expected_shape and image.size == np.prod(expected_shape):
                                image = image.reshape(expected_shape)
                        
                        X[v][i] = image
                        
                    except Exception as e:
                        print(f"[ERROR] Cargando imagen {full_path}: {e}")
                        # Llenar con ceros en caso de error
                        if colormode == "grayscale":
                            X[v][i] = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)
                        else:
                            X[v][i] = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.float32)

            # FIXED: Always return the correct format based on number of images
            if num_images == 1:
                # Solo una imagen por entrada (para modelos individuales en XAI)
                yield X[0]  # Devolver solo el array, no una lista
            else:
                # Múltiples imágenes (para modelos ensemble)
                # Asegurar que devolvemos exactamente el número de arrays que el modelo espera
                yield X  # Devolver lista de arrays
def load_image_normalize_grayscale(images,dir,img_shape,outdir,fit_data = True,list_values = []):
    """
    Función de normalización adaptada para número variable de proyecciones
    """
    if fit_data:
        img_mean = []
        img_std = []
        for img in images:
            # Procesar solo las imágenes (excluyendo label y posible weight)
            num_images = len(img) - 2 if len(img) > 3 else len(img) - 1
            for i in img[:num_images]:
                image = img_to_array(load_img(os.path.join(dir,i), color_mode='grayscale', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0
                img_mean.append(np.mean(image))
                img_std.append(np.std(image))

        img_mean = np.mean(img_mean)
        img_std = np.mean(img_std)
        
        print(img_mean, img_std)

    else:
        img_mean = list_values[0]
        img_std = list_values[1]

    img_channel = np.full((img_shape[0],img_shape[1],1), img_mean)
    print(np.shape(img_channel))

    means = []
    stds = []

    for img in images:
        num_images = len(img) - 2 if len(img) > 3 else len(img) - 1
        for i in img[:num_images]:
            image = img_to_array(load_img(os.path.join(dir,i), color_mode='grayscale', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0

            image = (image - img_channel) / img_std

            means.append(np.mean(image))
            stds.append(np.std(image))

            if not os.path.exists(os.path.dirname(os.path.join(outdir,i))):
                os.makedirs(os.path.dirname(os.path.join(outdir,i)))
            cv.imwrite(os.path.join(outdir,i),image)

    print('Media final', np.mean(means), np.mean(stds))
    return [img_mean, img_std]

def load_image_normalize_rgb(images,dir,img_shape,outdir,fit_data = True,list_values = []):
    """
    Función de normalización RGB adaptada para número variable de proyecciones
    """
    if fit_data:
        r_mean = []
        g_mean = []
        b_mean = []
        r_std = []
        g_std = []
        b_std = []
        for img in images:
            num_images = len(img) - 2 if len(img) > 3 else len(img) - 1
            for i in img[:num_images]:
                image = img_to_array(load_img(os.path.join(dir,i), color_mode='rgb', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0
                r, g, b = np.split(image,3,axis=2)
                r_mean.append(np.mean(r))
                g_mean.append(np.mean(g))
                b_mean.append(np.mean(b))
                r_std.append(np.std(r))
                g_std.append(np.std(g))
                b_std.append(np.std(b))

        r_mean = np.mean(r_mean)
        r_std = np.mean(r_std)
        g_mean = np.mean(g_mean)
        g_std = np.mean(g_std)
        b_mean = np.mean(b_mean)
        b_std = np.mean(b_std)

        print(r_mean,r_std)
        print(g_mean,g_std)
        print(b_mean,b_std)

    else:
        r_mean = list_values[0]
        r_std = list_values[1]
        g_mean = list_values[2]
        g_std = list_values[3]
        b_mean = list_values[4]
        b_std = list_values[5]

    r_channel = np.full((img_shape[0],img_shape[1],1), r_mean)
    g_channel = np.full((img_shape[0],img_shape[1],1), g_mean)
    b_channel = np.full((img_shape[0],img_shape[1],1), b_mean)
    
    print(np.shape(r_channel),np.shape(g_channel),np.shape(b_channel))

    means = []
    stds = []

    for img in images:
        num_images = len(img) - 2 if len(img) > 3 else len(img) - 1
        for i in img[:num_images]:
            image = img_to_array(load_img(os.path.join(dir,i), color_mode='rgb', target_size=img_shape, interpolation='bilinear')).astype(np.float32) / 255.0

            r, g, b = np.split(image,3,axis=2)
            r = (r - r_channel) / r_std
            g = (g - g_channel) / g_std
            b = (b - b_channel) / b_std

            image = np.stack((r,g,b),axis=2)
            image = np.squeeze(image,axis=3)

            means.append(np.mean(image))
            stds.append(np.std(image))

            if not os.path.exists(os.path.dirname(os.path.join(outdir,i))):
                os.makedirs(os.path.dirname(os.path.join(outdir,i)))
            
            cv.imwrite(os.path.join(outdir,i),image)

    print('Media final', np.mean(means), np.mean(stds))
    return [r_mean,r_std,g_mean,g_std,b_mean,b_std]

def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

def bin(age, age_list):
    unique_list = []
    for i in np.arange(np.max(age_list)+1):
        if i not in unique_list:
            unique_list.append(i)

    return unique_list.index(age)

def inverse(x):
    v = np.float32(1/x)
    if v == np.Inf:
        v = np.float32(1)
    return v

def calculate_weights(sample_df):
    age_list = sample_df['Age'].to_list()
    bin_index_per_label = [bin(label,age_list) for label in age_list]
    
    N_ranges = max(bin_index_per_label) + 1
    num_samples_of_bin = dict(Counter(bin_index_per_label))

    emp_label_dist = [num_samples_of_bin.get(i,0) for i in np.arange(N_ranges)]
   
    lds_kernel = cv.getGaussianKernel(5,2).flatten()
    eff_label_dist = sci.convolve1d(np.array(emp_label_dist), weights=lds_kernel, mode='constant')
    
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    
    weights = [inverse(x) for x in eff_num_per_label]

    sample_df['Weight'] = weights

def train_test_split(sample_df, size):
    dic_aux = {}
    sub_df = sample_df
    column = 'Number'
    column_number = 0

    range = np.unique(np.array(sub_df[column].to_list()))

    div = int(len(range) * size)
    if div == 0:
        div = 1

    np.random.shuffle(range)

    for i in range[:div]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Train'

    for i in range[div:]:
        for name in sub_df[sub_df[column] == i].values:
            dic_aux[name[column_number]] = 'Test'

    set_list = []

    for i in sample_df[column].values:
        set_list.append(dic_aux[i])
    
    sample_df.insert(0,'Set', set_list, True)

    train_df = sample_df[sample_df["Set"] == 'Train']
    train_df = train_df.drop(columns=["Set"])

    test_df = sample_df[sample_df["Set"] == 'Test']
    test_df = test_df.drop(columns=["Set"])

    return train_df, test_df

def range_of_age(n,ranges):
    for i,r in enumerate(ranges):
        if n >= r[0] and n <= r[1]:
            return i

def precision_by_range(y_true, y_pred, ranges, metric='mae'):

    prec_dic = {}
    for r in np.arange(len(ranges)):
        prec_dic[r] = [[],[],[]]

    for yt, yp in zip(y_true, y_pred):
        r_age = range_of_age(yt, ranges)

        if metric == 'mae':
            dif = abs(yt - yp)
            metric_name = 'MAE'
        elif metric == 'mse':
            dif = (yt - yp) * (yt - yp)
            metric_name = 'MSE'
        elif metric == 'rmse':
            dif = (yt - yp) * (yt - yp)
            metric_name = 'RMSE'
        prec_dic[r_age][0].append(dif)
        prec_dic[r_age][1].append(yp)
        prec_dic[r_age][2].append(yt)
    
    df_precision = pd.DataFrame()

    prec_dic_filter = {}
    for k in prec_dic.keys():
        if len(np.array(prec_dic[k][1])) != 0:
            prec_dic_filter[k] = prec_dic[k]

    df_precision['Range'] = [ranges[k] for k in prec_dic_filter.keys()]
    df_precision['N values'] = [len(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    df_precision['T values Mean'] = [np.mean(np.array(prec_dic_filter[k][2])) for k in prec_dic_filter.keys()]
    df_precision['Values Mean'] = [np.mean(np.array(prec_dic_filter[k][1])) for k in prec_dic_filter.keys()]
    
    if metric_name == "RMSE":
        df_precision[metric_name] = [np.sqrt(np.mean(np.array(prec_dic_filter[k][0]))) for k in prec_dic_filter.keys()]
    else:
        df_precision[metric_name] = [np.mean(np.array(prec_dic_filter[k][0])) for k in prec_dic_filter.keys()]

    return df_precision

def show_stats(true, pred, metric_range='mae'):
    stats_mae = abs(true-pred)
    stats_mse = (true-pred)*(true-pred)
    measures = [stats_mae, stats_mse, stats_mse]

    df_stats = pd.DataFrame()
    df_stats['Metric'] = ['MAE' ,'MSE', 'RMSE']
    df_stats['Mean:'] = [np.mean(stats_mae), np.mean(stats_mse), np.sqrt(np.mean(stats_mse))]

    ranges = [roa.ranges_todd,roa.ranges_5,roa.ranges_3]
    df_precision_complete_list = []
    for range in ranges:
        df_precision = precision_by_range(true, pred, range, metric_range)
        if metric_range == 'mae':
            print('Mean of means:', np.mean(df_precision['MAE'].to_numpy()))
        elif metric_range == 'mse':
            print('Mean of means:', np.mean(df_precision['MSE'].to_numpy()))
        elif metric_range == 'rmse':
            print('Mean of means:', np.mean(df_precision['RMSE'].to_numpy()))
        print(df_precision.to_string())
        df_precision_complete_list.append(df_precision)

    
    df_precision_complete = pd.concat(df_precision_complete_list)
    print(df_stats.to_string())

    return df_stats, df_precision_complete


