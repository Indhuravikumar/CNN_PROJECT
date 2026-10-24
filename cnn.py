import os, glob
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models


url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
tf.keras.utils.get_file("catsdogs.zip", url, extract=True, cache_dir=".", cache_subdir="data")
data_dir = glob.glob("data/**/PetImages", recursive=True)[0]


for cls in ["Cat", "Dog"]:
    folder = os.path.join(data_dir, cls)
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        try:
            img = Image.open(p)
            img.load()
            img.convert("RGB").save(p, "JPEG")   # force 3-channel RGB JPEG
        except Exception:
            os.remove(p)


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42,
    image_size=(128, 128), batch_size=32, label_mode="binary")
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=(128, 128), batch_size=32, label_mode="binary")
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=2, validation_data=test_ds)

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc * 100:.2f}%")