import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


training_dir = "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\cashutosh\\gender-classification-dataset\\versions\\1\\Training"
validation_dir = "C:\\Users\\USER\\.cache\\kagglehub\\datasets\\cashutosh\\gender-classification-dataset\\versions\\1\\Validation"

batch_size = 32
image_size = (128, 128)

# ************************************************************************************************
# One option that wont let us use data augmentation
# train_dataset = keras.preprocessing.image_dataset_from_directory(training_dir, 
#                                                                     image_size=image_size, 
#                                                                     batch_size=batch_size, 
#                                                                     label_mode='int')

# val_dataset = keras.preprocessing.image_dataset_from_directory(validation_dir, 
#                                                                   image_size=image_size, 
#                                                                   batch_size=batch_size, 
#                                                                   label_mode='int')


# normalization_layer = keras.layers.Rescaling(1./255)
# train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
# val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))


# ************************************************************************************************
# Option 2 with Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save("C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\gender_classifier_model.h5")