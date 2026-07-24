import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        if use_batch_norm:
            shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def resnet18(input_shape=(224, 224, 3), return_embeddings=True):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    
    if return_embeddings:
        embeddings = x
        output = layers.Dense(512)(embeddings)  # Optional: Modify embedding dimensions
    else:
        embeddings = None
        output = x

    model = models.Model(inputs, output)

    return model, models.Model(inputs, embeddings) if return_embeddings else None