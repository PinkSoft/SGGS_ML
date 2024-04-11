import os
from datetime import datetime
import random
#from glob import glob
from pathlib import Path
from dotenv import load_dotenv
#import requests
#from zipfile import ZipFile
#import glob
from dataclasses import dataclass, field

#import numpy as np
#import cv2
import tensorflow as tf
import keras_cv
import pandas as pd
import keras
from keras import layers

import image_util as iu
import Models as models

#import matplotlib.pyplot as plt

num = 40

def system_config(SEED_VALUE):
    # Set python `random` seed.
    # Set `numpy` seed
    # Set `tensorflow` seed.
    random.seed(SEED_VALUE)
    tf.keras.utils.set_random_seed(SEED_VALUE)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'     
    os.environ['TF_USE_CUDNN'] = "true"
    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_path():
    load_dotenv()
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.getenv("DATA_PATH_512")

segments = {
    0: 'Lumen',  # Lumen
    1: 'Stenosis',  # Stenose
    2: 'Multi',  # Lumen and stenose
    }

@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE: tuple = (512, 512)
    BATCH_SIZE: int = 4 # 16
    NUM_CLASSES: int = 2
    BRIGHTNESS_FACTOR: float = 0.2
    CONTRAST_FACTOR: float = 0.2
    MASK: str = segments[1]

@dataclass(frozen=True)
class TrainingConfig:
    BACKBONE: str = "resnet50_v2_imagenet"
    WEIGHTS: str = 'imagenet'
    MODEL: str = "Unet"
    EPOCHS: int = 100  # 100 # 35
    LEARNING_RATE: float = 1e-4 #1e-4
    TRAIN_NO : str = "00"
    CKPT_DIR: str = os.path.join("/home/bp/Development/SGGS_ML", "checkpoints_"+"_".join(MODEL.split("_")[:2]),
                                        "_".join(MODEL.split("_")[:2])+"_"+TRAIN_NO+".h5")
    LOGS_DIR: str = os.path.join("/home/bp/Development/SGGS_ML/logs", "_".join(MODEL.split("_")[:2]))
    HIST_DIR: str = os.path.join("/home/bp/Development/SGGS_ML/hist", "_".join(MODEL.split("_")[:2]),
                                        ""+"_".join(MODEL.split("_")[:2])+"_"+TRAIN_NO)
    
    #print(CKPT_DIR)
    #print(LOGS_DIR)
    #print(HIST_DIR)

def unpackage_inputs(inputs):
    images = inputs["images"]
    segmentation_masks = inputs["segmentation_masks"]
    return images, segmentation_masks

# Dictionary mapping class IDs to colors.
id2color = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # lumen
    2: (0, 0, 255),  # stenose
}

def mean_iou(y_true, y_pred):

    # Get total number of classes from model output.
    num_classes = y_pred.shape[-1]

    y_true = tf.squeeze(y_true, axis=-1)

    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, axis=-1)
    y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes, axis=-1)

    # Intersection: |G ∩ P|. Shape: (batch_size, num_classes)
    intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))

    # Total Sum: |G| + |P|. Shape: (batch_size, num_classes)
    total = tf.math.reduce_sum(y_true, axis=(1, 2)) + tf.math.reduce_sum(y_pred, axis=(1, 2))

    union = total - intersection

    is_class_present =  tf.cast(tf.math.not_equal(total, 0), dtype=tf.float32)
    num_classes_present = tf.math.reduce_sum(is_class_present, axis=1)

    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.math.reduce_sum(iou, axis=1) / num_classes_present

    # Compute the mean across the batch axis. Shape: Scalar
    return tf.math.reduce_mean(iou)

def get_callbacks(
    train_config,
    monitor="val_mean_iou",
    mode="max",
    save_weights_only=True,
    save_best_only=True,
):

    # Initialize tensorboard callback for logging.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=train_config.LOGS_DIR,
        histogram_freq=20,
        write_graph=False,
        update_freq="epoch",
    )


    # Update file path if saving best model weights.
    if save_weights_only:
        checkpoint_filepath =  "/home/bp/Development/SGGS_ML/checkpoints/Unet/S_" + TrainingConfig.MODEL + "_" + str(num) + ".h5"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=save_weights_only,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=1,
    )

    return [tensorboard_callback, model_checkpoint_callback]

def save_results(df, path):
    #print(dataframe)
    # convert the history.history dict to a pandas DataFrame:
    df = pd.DataFrame(df)
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # save to csv:
    df_csv_file = path + '_' + dt_string +'.csv'
    with open(df_csv_file, mode='w', encoding="utf-8") as f:
        df.to_csv(f)

if __name__ == "__main__":

    learning_rate = (5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6)
    batch_size = (4, 8, 12, 16, 20, 24, 28, 32) # 16
    system_config(SEED_VALUE=42)
    train_config = TrainingConfig()
    data_config = DatasetConfig()
    PATH = load_path()
    result_df = pd.DataFrame(columns=['index', 'batch', 'lr',
                                      'train_loss', 'train_acc','train_mean_iou', 
                                      'val_loss', 'val_acc', 'val_mean_iou',
                                      'test_loss', 'test_acc', 'test_mean_iou' 
    ])
    train_ds, valid_ds, test_ds = iu.get_data_sets(path=PATH, mask_set=data_config.MASK)

    augment_fn = tf.keras.Sequential(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandomBrightness(factor=data_config.BRIGHTNESS_FACTOR,
                                         value_range=(0, 255)),
        keras_cv.layers.RandomContrast(factor=data_config.CONTRAST_FACTOR,
                                       value_range=(0, 255)),
    ]
    )
    
    for batch in batch_size:
        for lr in learning_rate:
            train_dataset = (
                        train_ds.shuffle(batch) #data_config.BATCH_SIZE)
                        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(batch) #data_config.BATCH_SIZE)
                        .map(unpackage_inputs)
                        .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

            valid_dataset = (
                        valid_ds.batch(batch) #data_config.BATCH_SIZE)
                        .map(unpackage_inputs)
                        .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

            test_dataset = (
                        test_ds.batch(batch) #data_config.BATCH_SIZE)
                        .map(unpackage_inputs)
                        .prefetch(buffer_size=tf.data.AUTOTUNE)
            )

            # Build model.        
            backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
                preset=train_config.MODEL,
                input_shape=data_config.IMAGE_SIZE + (3,),
                load_weights=True,
            )

            model = models.transunet_2d(
                (DatasetConfig.IMAGE_SIZE[0], DatasetConfig.IMAGE_SIZE[1], 3),
                filter_num=[64, 128, 256, 512],
                n_labels=12,
                stack_num_down=2,
                stack_num_up=2,
                embed_dim=768,
                num_mlp=3072,
                num_heads=12,
                num_transformer=12,
                activation="ReLU",
                mlp_activation="GELU",
                output_activation="Softmax",
                batch_norm=True,
                pool=True,
                unpool="bilinear",
                backbone=backbone,
                #weights='imagenet', 
                name="transunet",
            )
            # Get callbacks.
            callbacks = get_callbacks(train_config)
            # Define Loss.
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            # Compile model.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(lr), # train_config.LEARNING_RATE),
                loss=loss_fn,
                metrics=["accuracy", mean_iou],
            )

            try:
                # Train the model, doing validation at the end of each epoch.
                history = model.fit(
                    train_dataset,
                    epochs=train_config.EPOCHS,
                    validation_data=valid_dataset,
                    callbacks=callbacks
                )
                save_results(history.history, train_config.HIST_DIR + '_' + str(num))
                model.load_weights( "/home/bp/Development/SGGS_ML/checkpoints/Unet/S_" + TrainingConfig.MODEL + "_" + str(num) + ".h5") #train_config.CKPT_DIR)    
                evaluate = model.evaluate(test_dataset)
                #print("Test loss:", evaluate[2])
                hist_df = pd.DataFrame(history.history)
                max_index = hist_df['val_mean_iou'].idxmax()
                #print(max_index)
                new_row_data = {
                    'index' : num,
                    'batch' : batch,
                    'lr'    : lr,
                    'train_loss':  hist_df.iloc[max_index]['loss'],
                    'train_acc':  hist_df.iloc[max_index]['accuracy'],
                    'train_mean_iou':  hist_df.iloc[max_index]['mean_iou'],
                    'val_loss':  hist_df.iloc[max_index]['val_loss'],
                    'val_acc':  hist_df.iloc[max_index]['val_accuracy'],
                    'val_mean_iou':  hist_df.iloc[max_index]['val_mean_iou'],
                    'test_loss': evaluate[0],
                    'test_acc': evaluate[1],
                    'test_mean_iou': evaluate[2]
                    }

            except Exception as e:
                print("An exception occurred")
                print(e)
                new_row_data = {
                    'index' : num,
                    'batch' : batch,
                    'lr'    : lr,
                    'train_loss':0,
                    'train_acc': 0,
                    'train_mean_iou': 0,
                    'val_loss':0,
                    'val_acc': 0,
                    'val_mean_iou': 0,
                    'test_loss': 0,
                    'test_acc': 0,
                    'test_mean_iou' : 0
                    }
            #result_df = result_df.append(new_row_data, ignore_index=True)
            result_df = pd.concat([result_df, pd.DataFrame([new_row_data])], ignore_index=True)
            save_results(result_df, "/home/bp/Development/SGGS_ML/results/S_" + str(num) + "_" + TrainingConfig.MODEL)
            num += 1
            #print('Batchsize: ',batch, ' LearningRate: ', lr)
    save_results(result_df, "/home/bp/Development/SGGS_ML/S_results" + "_" + TrainingConfig.MODEL)