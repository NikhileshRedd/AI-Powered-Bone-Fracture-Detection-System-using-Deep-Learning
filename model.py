import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    r"C:\Users\ngond\OneDrive\Pictures\data_science\X_Ray\train",
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary",
    shuffle=True
)

# Testing data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    r"C:\Users\ngond\OneDrive\Pictures\data_science\X_Ray\test",
    
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Display class labels
print("Class Indices:", train_data.class_indices)

# Build CNN model
model = Sequential([
    Input(shape=(64, 64, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Display model summary
model.summary()

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save("model.keras")

print("Model saved successfully as model.keras")