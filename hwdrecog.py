import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Preprocess the MNIST Dataset
print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing pixel values to a range of 0 to 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# 2. Build the Neural Network Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
print("\nStarting Training...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 5. Evaluate and Visualize Training Progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# 6. Test the model on sample images (Requirement's Last Line)
print("\nTesting on sample images...")
predictions = model.predict(X_test)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    # Display Predicted vs Actual labels
    plt.title(f"Pred: {np.argmax(predictions[i])}\nActual: {y_test[i]}")
    plt.axis('off')
plt.suptitle("Sample Image Predictions")
plt.show()

# Save the model for future use
model.save('my_model.h5')