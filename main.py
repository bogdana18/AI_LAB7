import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Завантаження даних MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормалізація (перетворення значень пікселів у діапазон [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Розширення розмірності (необхідно для згорткових мереж)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Побудова моделі згорткової нейронної мережі
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 класів (0-9)
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Оцінка точності
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Точність на тестових даних: {test_acc:.4f}')
