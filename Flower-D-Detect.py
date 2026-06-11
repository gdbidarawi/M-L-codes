import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Dataset folders:
# flowers/
#   train/
#      healthy/
#      powdery_mildew/
#      leaf_spot/
#   validation/

IMG_SIZE = (224, 224)
BATCH = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2
)

val_gen = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(
    "flowers/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

valid = val_gen.flow_from_directory(
    "flowers/validation",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical"
)

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable = False

x = GlobalAveragePooling2D()(base.output)
output = Dense(train.num_classes, activation="softmax")(x)

model = Model(base.input, output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=valid, epochs=10)

model.save("flower_disease_model.h5")