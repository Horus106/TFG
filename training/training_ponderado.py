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

def main(train_path, validation_path, test_path, data_path, checkpoint_filepath, 
         batch_size, color_mode, epochs, output, resnet, seed, img_shape, depth, 
         cnn, verbose_level, pretrained_model,
         mae_x, mae_y, mae_z):

    mse = []
    r2 = []
    rmse = []
    mae = []
    name = []
    fold = 1
    ################################################################################
    cnn_0 = cnn(0,img_shape[0],img_shape[1],depth)
    cnn_1 = cnn(1,img_shape[0],img_shape[1],depth)
    cnn_2 = cnn(2,img_shape[0],img_shape[1],depth)

    # Cálculo de pesos normalizados en base a MAE
    inv_mae_sum = 1/mae_x + 1/mae_y + 1/mae_z
    w_x = (1/mae_x) / inv_mae_sum
    w_y = (1/mae_y) / inv_mae_sum
    w_z = (1/mae_z) / inv_mae_sum

    print(f"[INFO] Pesos de fusión (X,Y,Z): {w_x:.4f}, {w_y:.4f}, {w_z:.4f}")

    # Media ponderada
    from tensorflow.keras.layers import Lambda
    import tensorflow.keras.backend as K

    x = Lambda(lambda tensors: w_x*tensors[0] + w_y*tensors[1] + w_z*tensors[2])(
        [cnn_0.output, cnn_1.output, cnn_2.output]
    )

    model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

    optimizer = Adam()

    model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])
    model.summary()

    weights = model.get_weights()

    if resnet:
        print('FINE TUNNING')
        cnn_0 = fn.create_CNN_Resnet(0,img_shape[0],img_shape[1],depth,True)
        cnn_1 = fn.create_CNN_Resnet(1,img_shape[0],img_shape[1],depth,True)
        cnn_2 = fn.create_CNN_Resnet(2,img_shape[0],img_shape[1],depth,True)
        x = Average()([cnn_0.output, cnn_1.output, cnn_2.output])

        model_ft = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

        optimizer_ft = Adam(1e-5)

        model_ft.compile(loss=MeanAbsoluteError(), optimizer=optimizer_ft, metrics=['mae'])
        model_ft.summary()

    ################################################################################

    print('Data:', color_mode)
    print('Batch:', batch_size)
    print('Epochs:', epochs)
    print('Seed:', seed)
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

    train = fn.get_images_list(train_df,color_mode)
    validation = fn.get_images_list(validation_df,color_mode,weights=False)
    test = fn.get_images_list(test_df,color_mode,weights=False)
    test_not_augment = fn.get_images_list(test_df,color_mode,augment=False,weights=False)
    datagen = ImageDataGenerator()
    print("[INFO]: Starting image load process...")
    datagen_train = fn.image_generator(train, data_path, batch_size, datagen, img_shape, colormode=color_mode)
    print("[INFO]: Train data loaded...")
    datagen_val = fn.image_generator(validation, data_path, batch_size, datagen, img_shape, colormode=color_mode, weights=False)
    print("[INFO]: Validation data loaded...")
    datagen_test = fn.image_generator(test, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
    datagen_test_not_augment = fn.image_generator(test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
    print("[INFO]: Test data loaded...")

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
        callbacks=[early_stop_callback, model_checkpoint_callback]
    )

    history_df = pd.DataFrame.from_dict(history.history)

    model.load_weights(checkpoint_filepath)

    prediction = model.predict(datagen_test, verbose=verbose_level, steps=len(test))
    true = np.array([float(n) for n in test[:,3]])
    pred = prediction.flatten()

    df_stats_augment, df_precision_augment = fn.show_stats(true,pred)

    name.append("Prediction Augment")
    mse.append(mean_squared_error(true, pred))
    r2.append(r2_score(true, pred))
    rmse.append(mean_squared_error(true, pred, squared= False))
    mae.append(mean_absolute_error(true, pred))

    prediction_not_augment = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
    true = np.array([float(n) for n in test_not_augment[:,3]])
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

    if not resnet:
        if color_mode == 'rgb':
            output_subfolder = 'model_rgb_panoramacnn'
            output_folder = os.path.join(output, output_subfolder)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output, output_subfolder, 'model_rgb_panoramacnn')
            model.save_weights(output_path)

        elif color_mode == 'grayscale':
            output_subfolder = 'model_gray_panoramacnn'
            output_folder = os.path.join(output, output_subfolder)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output, output_subfolder, 'model_gray_panoramacnn')

        history_df.to_csv(os.path.join(output_folder, "history.csv"), sep = ";")
        df_stats_augment.to_csv(os.path.join(output_folder, "stats_augment.csv"), sep = ";")
        df_precision_augment.to_csv(os.path.join(output_folder, "precision_augment.csv"), sep = ";")
        df_stats_not_augment.to_csv(os.path.join(output_folder, "stats_not_augment.csv"), sep = ";")
        df_precision_not_augment.to_csv(os.path.join(output_folder, "precision_not_augment.csv"), sep = ";")

    else:
        output_subfolder = "model_rgb_resnet"
        output_folder = os.path.join(output, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'model_rgb_resnet')
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

        prediction = model_ft.predict(datagen_test, verbose=verbose_level, steps=len(test))
        true = np.array([float(n) for n in test[:,3]])
        pred = prediction.flatten()

        df_stats_augment_ft, df_precision_augment_ft = fn.show_stats(true,pred)

        name.append("Prediction Augment FT")
        mse.append(mean_squared_error(true, pred))
        r2.append(r2_score(true, pred))
        rmse.append(mean_squared_error(true, pred, squared= False))
        mae.append(mean_absolute_error(true, pred))

        prediction_not_augment = model_ft.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
        true = np.array([float(n) for n in test_not_augment[:,3]])
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
        output_subfolder = "model_rgb_resnet_ft"
        output_folder = os.path.join(output, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'model_rgb_resnet_ft')
        model_ft.save_weights(output_path)

        history_df_ft.to_csv(os.path.join(output_folder, "history_ft.csv"), sep = ";")
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
    if resnet:
        result_code = "RESNET50"
    else:
        result_code = "PANORAMA-CNN"

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
        args["mae_x"],
        args["mae_y"],
        args["mae_z"])