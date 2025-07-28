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

from datetime import datetime

import functions as fn

import argparse
import logging

import json


################################################################################

# CORRECCIÓN: Mover la clase WeightMonitor fuera de la función main
class WeightMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model_proj1, model_proj2):
        self.model_proj1 = model_proj1
        self.model_proj2 = model_proj2
        self.initial_weights_1 = [w.copy() for w in model_proj1.get_weights()]
        self.initial_weights_2 = [w.copy() for w in model_proj2.get_weights()]
        self.sample_data = None
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Verificar cada 5 épocas
            # Verificar si los pesos están cambiando
            current_weights_1 = self.model_proj1.get_weights()
            current_weights_2 = self.model_proj2.get_weights()
            
            change_1 = np.mean([np.mean(np.abs(cw - iw)) for cw, iw in zip(current_weights_1, self.initial_weights_1)])
            change_2 = np.mean([np.mean(np.abs(cw - iw)) for cw, iw in zip(current_weights_2, self.initial_weights_2)])
            
            print(f"[DEBUG] Época {epoch}: Cambio pesos modelo 1: {change_1:.6f}, modelo 2: {change_2:.6f}")
            
            # Verificar predicciones individuales - CORREGIDO: usar las entradas correctas
            if self.sample_data is not None:
                try:
                    # La sample_data contiene [input1, input2] para el ensemble
                    # Cada modelo individual necesita su entrada correspondiente
                    pred_1 = self.model_proj1.predict(self.sample_data[0][:1], verbose=0)
                    pred_2 = self.model_proj2.predict(self.sample_data[1][:1], verbose=0)
                    print(f"[DEBUG] Predicción modelo 1: {pred_1[0][0]:.4f}, modelo 2: {pred_2[0][0]:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Error en predicción individual: {e}")
                    # Solo mostrar cambios de pesos si hay error en predicciones
                    pass


def main(train_path, validation_path, test_path, data_path, checkpoint_filepath, 
         batch_size, color_mode, epochs, output, resnet, seed, img_shape, depth, 
         cnn, verbose_level, pretrained_model, projection_1, projection_2,
         mae_x, mae_y, mae_z):

    mse = []
    r2 = []
    rmse = []
    mae = []
    name = []
    fold = 1
    
    # Mapeo de proyecciones a índices
    projection_map = {'X': 0, 'Y': 1, 'Z': 2}
    proj1_idx = projection_map[projection_1]
    proj2_idx = projection_map[projection_2]

    # Calcular pesos de fusión
    mae_dict = {'X': mae_x, 'Y': mae_y, 'Z': mae_z}
    mae_1 = mae_dict[projection_1]
    mae_2 = mae_dict[projection_2]

    inv_sum = 1 / mae_1 + 1 / mae_2
    w_1 = (1 / mae_1) / inv_sum
    w_2 = (1 / mae_2) / inv_sum

    print(f"[INFO] Pesos de fusión ({projection_1}, {projection_2}): {w_1:.4f}, {w_2:.4f}")
    
    print(f'Using ensemble of projections: {projection_1} and {projection_2}')
    
    # CRÍTICO: Limpiar el estado de TensorFlow antes de crear modelos
    tf.keras.backend.clear_session()
    
    # Crear los modelos con índices diferentes para evitar conflictos
    print("[INFO] Creando modelo para primera proyección...")
    cnn_proj1 = cnn(proj1_idx, img_shape[0], img_shape[1], depth)
    
    print("[INFO] Creando modelo para segunda proyección...")
    cnn_proj2 = cnn(proj2_idx + 10, img_shape[0], img_shape[1], depth)  # +10 para garantizar unicidad

    # Crear el ensemble
    x = Lambda(lambda tensors: w_1 * tensors[0] + w_2 * tensors[1], name='ensemble_weighted')(
        [cnn_proj1.output, cnn_proj2.output]
    )
    model = Model(inputs=[cnn_proj1.input, cnn_proj2.input], outputs=x, name='ensemble_model')

    # CRÍTICO: Usar diferentes learning rates para cada parte del modelo si es necesario
    optimizer = Adam(learning_rate=1e-4)  # Learning rate más conservador
    model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])
    
    print("[INFO] Modelo ensemble creado:")
    model.summary()

    # Verificar que ambos submodelos son diferentes
    print(f"[DEBUG] Pesos del modelo 1: {len(cnn_proj1.get_weights())}")
    print(f"[DEBUG] Pesos del modelo 2: {len(cnn_proj2.get_weights())}")
    
    # Guardar pesos iniciales para verificación
    initial_weights_1 = [w.copy() for w in cnn_proj1.get_weights()]
    initial_weights_2 = [w.copy() for w in cnn_proj2.get_weights()]

    weights = model.get_weights()

    # CORRECCIÓN: Crear instancia del monitor antes de usarlo
    weight_monitor = WeightMonitor(cnn_proj1, cnn_proj2)

    if resnet:
        print('FINE TUNING')
        tf.keras.backend.clear_session()  # Limpiar antes del fine-tuning
        
        cnn_proj1_ft = fn.create_CNN_Resnet(proj1_idx, img_shape[0], img_shape[1], depth, True)
        cnn_proj2_ft = fn.create_CNN_Resnet(proj2_idx + 20, img_shape[0], img_shape[1], depth, True)  # +20 para unicidad
        
        x = Lambda(lambda tensors: w_1 * tensors[0] + w_2 * tensors[1], name='ensemble_weighted_ft')(
            [cnn_proj1_ft.output, cnn_proj2_ft.output]
        )
        model_ft = Model(inputs=[cnn_proj1_ft.input, cnn_proj2_ft.input], outputs=x, name='ensemble_model_ft')

        optimizer_ft = Adam(learning_rate=1e-5)
        model_ft.compile(loss=MeanAbsoluteError(), optimizer=optimizer_ft, metrics=['mae'])
        model_ft.summary()

    ################################################################################

    print('Data:', color_mode)
    print('Batch:', batch_size)
    print('Epochs:', epochs)
    print('Seed:', seed)
    print('Projections:', projection_1, 'and', projection_2)
    if resnet:
        print('RESNET50')
    else:
        print('PANORAMA-CNN')

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('[INFO]: Training -', dt_string)

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    train = fn.get_images_list(train_df, color_mode, projection_1, projection_2)
    validation = fn.get_images_list(validation_df, color_mode, projection_1, projection_2, weights=False)
    test_not_augment = fn.get_images_list(test_df, color_mode, projection_1, projection_2, augment=False, weights=False)

   

    
    datagen = ImageDataGenerator()

    # === NUEVO: Conjunto de test con augmentación ===
    test = fn.get_images_list(test_df, color_mode, projection_1, projection_2, augment=True, weights=False)
    datagen_test = fn.image_generator_fixed(test, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
    print("[INFO]: Starting image load process...")
    
    # Use the corrected image_generator function
    datagen_train = fn.image_generator_fixed(train, data_path, batch_size, datagen, img_shape, colormode=color_mode)
    print("[INFO]: Train data loaded...")
    datagen_val = fn.image_generator_fixed(validation, data_path, batch_size, datagen, img_shape, colormode=color_mode, weights=False)
    print("[INFO]: Validation data loaded...")
    datagen_test_not_augment = fn.image_generator_fixed(test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
    print("[INFO]: Test data loaded...")

    # Guardar muestra de datos para monitoreo - MEJORADO
    try:
        sample_batch = next(datagen_val)
        weight_monitor.sample_data = sample_batch[0]  # Las imágenes (ya es una lista con ambas entradas)
        print(f"[DEBUG] Sample data shapes: {[x.shape for x in sample_batch[0]]}")
    except Exception as e:
        print(f"[WARNING] No se pudo obtener muestra de validación para monitoreo: {e}")
        weight_monitor.sample_data = None

    if pretrained_model is not None:
        model.load_weights(pretrained_model)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(epochs*0.2))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=verbose_level)

    history = model.fit(
        datagen_train,
        validation_data=datagen_val,
        epochs=epochs,
        verbose=verbose_level,
        steps_per_epoch=(len(train) // batch_size),
        validation_steps=(len(validation) // batch_size),
        callbacks=[early_stop_callback, model_checkpoint_callback, weight_monitor]
    )

    history_df = pd.DataFrame.from_dict(history.history)

    model.load_weights(checkpoint_filepath)

    # === PREDICCIÓN CON AUGMENTACIÓN ===
    prediction = model.predict(datagen_test, verbose=verbose_level, steps=len(test))
    true_aug = np.array([float(n) for n in test[:,2]])
    pred_aug = prediction.flatten()

    df_stats_augment, df_precision_augment = fn.show_stats(true_aug, pred_aug)

    name.append("Prediction Augment")
    mse.append(mean_squared_error(true_aug, pred_aug))
    r2.append(r2_score(true_aug, pred_aug))
    rmse.append(mean_squared_error(true_aug, pred_aug, squared=False))
    mae.append(mean_absolute_error(true_aug, pred_aug))

    prediction_not_augment = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
    true = np.array([float(n) for n in test_not_augment[:,2]])  # Ajustado índice porque ahora solo hay 2 imágenes + label
    pred = prediction_not_augment.flatten()

    df_stats_not_augment, df_precision_not_augment = fn.show_stats(true,pred)

    name.append("Prediction NOT Augment")
    mse.append(mean_squared_error(true, pred))
    r2.append(r2_score(true, pred))
    rmse.append(mean_squared_error(true, pred, squared= False))
    mae.append(mean_absolute_error(true, pred))

    kfold_stats_df = pd.DataFrame()
    kfold_stats_df['Name'] = name
    kfold_stats_df['MSE'] = mse
    kfold_stats_df['RMSE'] = rmse
    kfold_stats_df['MAE'] = mae
    kfold_stats_df['R2'] = r2

    print(kfold_stats_df.to_string())

    # Crear nombres de carpetas dinámicos basados en las proyecciones seleccionadas
    proj_suffix = f"{projection_1}{projection_2}"
    
    if not resnet:
        if color_mode == 'rgb':
            output_subfolder = f'model_rgb_panoramacnn_{proj_suffix}'
            output_folder = os.path.join(output, output_subfolder)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output, output_subfolder, f'model_rgb_panoramacnn_{proj_suffix}')
            model.save_weights(output_path)

        elif color_mode == 'grayscale':
            output_subfolder = f'model_gray_panoramacnn_{proj_suffix}'
            output_folder = os.path.join(output, output_subfolder)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output, output_subfolder, f'model_gray_panoramacnn_{proj_suffix}')

        history_df.to_csv(os.path.join(output_folder, "history.csv"), sep = ";")
        df_stats_not_augment.to_csv(os.path.join(output_folder, "stats_not_augment.csv"), sep = ";")
        df_precision_not_augment.to_csv(os.path.join(output_folder, "precision_not_augment.csv"), sep = ";")

    else:
        output_subfolder = f"model_rgb_resnet_{proj_suffix}"
        output_folder = os.path.join(output, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'model_rgb_resnet_{proj_suffix}')
        model.save_weights(output_path)
        history_df.to_csv(os.path.join(output_folder, "history.csv"), sep = ";")
        df_stats_not_augment.to_csv(os.path.join(output_folder, "stats_not_augment.csv"), sep = ";")
        df_precision_not_augment.to_csv(os.path.join(output_folder, "precision_not_augment.csv"), sep = ";")

        print('FINE TUNNING')

        model_ft.load_weights(checkpoint_filepath)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(epochs*0.2))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=verbose_level)

        history = model_ft.fit(
            datagen_train,
            validation_data=datagen_val,
            epochs=epochs,
            verbose=verbose_level,
            steps_per_epoch=(len(train) // batch_size),
            validation_steps=(len(validation) // batch_size),
            callbacks=[early_stop_callback, model_checkpoint_callback]
        )

        history_df_ft = pd.DataFrame.from_dict(history.history)

        model_ft.load_weights(checkpoint_filepath)

        # === PREDICCIÓN CON AUGMENTACIÓN (FINE-TUNING) ===
        prediction = model_ft.predict(datagen_test, verbose=verbose_level, steps=len(test))
        true_aug = np.array([float(n) for n in test[:,2]])
        pred_aug = prediction.flatten()

        df_stats_augment_ft, df_precision_augment_ft = fn.show_stats(true_aug, pred_aug)

        name.append("Prediction Augment FT")
        mse.append(mean_squared_error(true_aug, pred_aug))
        r2.append(r2_score(true_aug, pred_aug))
        rmse.append(mean_squared_error(true_aug, pred_aug, squared=False))
        mae.append(mean_absolute_error(true_aug, pred_aug))


        prediction_not_augment = model_ft.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
        true = np.array([float(n) for n in test_not_augment[:,2]])  # Ajustado índice
        pred = prediction_not_augment.flatten()

        df_stats_not_augment_ft, df_precision_not_augment_ft = fn.show_stats(true,pred)

        name.append("Prediction NOT Augment FT")
        mse.append(mean_squared_error(true, pred))
        r2.append(r2_score(true, pred))
        rmse.append(mean_squared_error(true, pred, squared=False))
        mae.append(mean_absolute_error(true, pred))

        kfold_stats_df = pd.DataFrame()
        kfold_stats_df['Name'] = name
        kfold_stats_df['MSE'] = mse
        kfold_stats_df['RMSE'] = rmse
        kfold_stats_df['MAE'] = mae
        kfold_stats_df['R2'] = r2

        print(kfold_stats_df.to_string())
        output_subfolder = f"model_rgb_resnet_ft_{proj_suffix}"
        output_folder = os.path.join(output, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'model_rgb_resnet_ft_{proj_suffix}')
        model_ft.save_weights(output_path)

        # === NUEVO: Guardar submodelos individuales ===
        print("[INFO] Guardando submodelos individuales...")

        # Extraer submodelo para cada vista directamente desde el modelo fine-tuned
        model_x = Model(inputs=model_ft.inputs[0], outputs=model_ft.get_layer("output_0").output)
        model_y = Model(inputs=model_ft.inputs[1], outputs=model_ft.get_layer("output_1").output)

        # Guardar pesos en la ruta esperada
        submodel_path_x = os.path.join(output_folder, "model_x")
        submodel_path_y = os.path.join(output_folder, "model_y")

        model_x.save_weights(submodel_path_x)
        print(f"[OK] Submodelo X guardado en: {submodel_path_x}")

        model_y.save_weights(submodel_path_y)
        print(f"[OK] Submodelo Y guardado en: {submodel_path_y}")

        history_df_ft.to_csv(os.path.join(output_folder, "history_ft.csv"), sep = ";")

        df_stats_augment_ft.to_csv(os.path.join(output_folder, "stats_augment_FT.csv"), sep = ";")
        df_precision_augment_ft.to_csv(os.path.join(output_folder, "precision_augment_FT.csv"), sep = ";")

        df_stats_not_augment_ft.to_csv(os.path.join(output_folder, "stats_not_augment_FT.csv"), sep = ";")
        df_precision_not_augment_ft.to_csv(os.path.join(output_folder, "precision_not_augment_FT.csv"), sep = ";")

    print('################################################################################')

    kfold_stats_df = pd.DataFrame()
    kfold_stats_df['Name'] = name
    kfold_stats_df['MSE'] = mse
    kfold_stats_df['RMSE'] = rmse
    kfold_stats_df['MAE'] = mae
    kfold_stats_df['R2'] = r2

    print('\n')
    print('Data:', color_mode)
    print('Batch:', batch_size)
    print('Epochs:', epochs)
    print('Seed:', seed)
    print('Projections:', projection_1, 'and', projection_2)
    if resnet:
        result_code = f"RESNET50_{proj_suffix}"
    else:
        result_code = f"PANORAMA-CNN_{proj_suffix}"

    print(result_code)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(dt_string)

    print(kfold_stats_df.to_string())
    kfold_stats_df.to_csv(os.path.join(output, "execution_results_{}.csv".format(result_code)), sep=";")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--parameters",
        help="JSON file with parameters configuration",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    json_file = open(args.parameters, "r")
    args = json.load(json_file)
    json_file.close()

    # Validar que las proyecciones estén especificadas
    if "projection_1" not in args or "projection_2" not in args:
        raise ValueError("projection_1 and projection_2 must be specified in the JSON parameters")
    
    # Validar que las proyecciones sean válidas
    valid_projections = ['X', 'Y', 'Z']
    if args["projection_1"] not in valid_projections or args["projection_2"] not in valid_projections:
        raise ValueError("Projections must be one of: X, Y, Z")
    
    # Validar que las proyecciones sean diferentes
    if args["projection_1"] == args["projection_2"]:
        raise ValueError("projection_1 and projection_2 must be different")

    if args["color_mode"] == 'rgb':
        img_shape = (36,108)
        depth = 3
        if args["resnet"]:
            cnn = fn.create_CNN_Resnet
        else:
            cnn = fn.create_CNN
    elif args["color_mode"] == 'grayscale':
        img_shape = (108,108)
        depth = 1
        cnn = fn.create_CNN
    else:
        raise RuntimeError("Color mode option unknown")

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])

    log_level = logging.WARNING
    if args["verbose"] == 0:
        log_level = logging.WARNING
    elif args["verbose"] == 1:
        log_level = logging.INFO
    elif args["verbose"] == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    main(
        args["train"],
        args["val"],
        args["test"],
        args["data"],
        args["checkpoint"],
        args["batch_size"], 
        args["color_mode"], 
        args["epochs"], 
        args["output"], 
        args["resnet"], 
        args["seed"], 
        img_shape, 
        depth, 
        cnn,
        args["verbose_keras"],
        args["pretrained_model"],
        args["projection_1"],
        args["projection_2"],
        args["mae_x"],
        args["mae_y"],
        args["mae_z"]
    )