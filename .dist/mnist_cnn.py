# mnist_cnn.py (or notebook cell)
"""
Task: Build CNN for MNIST, achieve >95% test accuracy, show predictions on 5 images.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess: scale and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)  # shape (N,28,28,1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels for training (Keras can also accept integer labels with sparse_categorical_crossentropy)
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 3. Model architecture
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint('mnist_cnn_best.h5', monitor='val_accuracy', save_best_only=True)
]

# 5. Train
history = model.fit(
    x_train, y_train_cat,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Ensure >95%
if test_acc < 0.95:
    print("Warning: accuracy < 95%. Consider more epochs, data augmentation, or different architecture.")

# 7. Visualize predictions on 5 sample images
import random
indices = random.sample(range(len(x_test)), 5)
preds = model.predict(x_test[indices])
pred_labels = np.argmax(preds, axis=1)

plt.figure(figsize=(12,4))
for i, idx in enumerate(indices):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[idx].squeeze(), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {pred_labels[i]}")
    plt.axis('off')
plt.show()

# Optionally save the model
# model.save('mnist_cnn_final.h5')
