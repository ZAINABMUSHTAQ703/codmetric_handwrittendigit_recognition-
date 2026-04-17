import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset (The "patterns" the machine learns from)
print("Accessing MNIST dataset...")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data Normalization
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define Model Architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training initiated...")
model.fit(X_train, y_train, epochs=5)

# Save the trained model to your directory
model.save('my_model.h5')
print("Model successfully saved as 'my_model.h5'")