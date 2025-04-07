import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Завантаження та підготовка даних
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Визначення назв класів
class_names = ['Футболка/топ', 'Штани', 'Светр', 'Сукня', 'Пальто',
               'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики']

# Побудова та компіляція моделі
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Навчання моделі
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)

# Виведення графіків точності та втрат
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точність на тренуванні')
plt.plot(history.history['val_accuracy'], label='Точність на валідації')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.title('Точність під час навчання')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Втрати на тренуванні')
plt.plot(history.history['val_loss'], label='Втрати на валідації')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()
plt.title('Втрати під час навчання')

plt.show()

# Оцінка моделі на тестових даних
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nТочність на тестових даних: {test_acc:.2f}')

# Виведення 40 зразків з передбаченнями
plt.figure(figsize=(10, 10))
for i in range(40):
    plt.subplot(8, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Прогноз: {class_names[np.argmax(model.predict(x_test[i:i+1]))]}\nРеальність: {class_names[y_test[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Збереження результатів у CSV
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
results = pd.DataFrame({'Реальна мітка': y_test, 'Передбачена мітка': predicted_labels})
results.to_csv('fashion_mnist_predictions.csv', index=False)
print("Результати збережено у файл 'fashion_mnist_predictions.csv'")
