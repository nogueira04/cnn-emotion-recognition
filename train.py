from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from data import train_generator, validation_generator

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-4, decay=1e-6),
              metrics="accuracy")

model_info = model.fit(
    train_generator, 
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.keras")