import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Завантаження бази даних зразків рукописних цифр MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Виведення перших 25 зображень з датасету (5x5 сітка)
plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Перші 25 зображень з MNIST", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Нормалізація (стандартизація) зображень: переведення пікселів у [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

#  4. Перетворення вхідних зображень 28x28 у вектори 784-елементні
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 5. Кодування міток у бінарному форматі (one-hot encoding, 10 класів)
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 6. Побудова архітектури нейронної мережі
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(784,)),                    # Вхідний шар: Flatten → вектор з 784 елементів
    tf.keras.layers.Dense(128, activation='relu', use_bias=True),  # Прихований шар: 128 нейронів, ReLU
    tf.keras.layers.Dense(10, activation='softmax', use_bias=True) # Вихідний шар: 10 нейронів, Softmax
])

# 7. Виведення структури побудованої моделі на консоль
print("\n----Структура нейронної мережі:")
model.summary()

# 8. Компіляція моделі з:
# - loss: категоріальна крос-ентропія
# - optimizer: Adam
# - метрика: точність (accuracy)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  9. Навчання моделі:
# - batch_size: 32
# - validation_split: 20%
# - epochs: 10 (можна змінити)
history = model.fit(
    x_train, y_train_cat,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=2
)

# 10. Запис результатів точності розпізнавання зразків по епохах
print("\n----Точність по епохах:")
for i, acc in enumerate(history.history['accuracy']):
    val_acc = history.history['val_accuracy'][i]
    print(f"Епоха {i+1}: Тренувальна = {acc:.4f}, Валідаційна = {val_acc:.4f}")

# 11. Розпізнавання одного тестового зображення та виведення на консоль
index = 0  # Індекс зображення
sample = x_test[index].reshape(1, 784)
prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)

print(f"\nРезультат розпізнавання:")
print(f"Передбачена цифра: {predicted_digit}")
print(f"Реальна цифра: {y_test[index]}")

# Відображення зображення
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Передбачено: {predicted_digit} | Реально: {y_test[index]}")
plt.axis('off')
plt.show()

# 12. Визначення кількості неправильно класифікованих зображень
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Масив індексів невірно класифікованих прикладів
incorrect_indices = np.where(predicted_classes != y_test)[0]
print(f"\nКількість неправильно класифікованих зображень: {len(incorrect_indices)}")

# 13. Виведення перших 5 таких зразків на консоль
plt.figure(figsize=(12, 4))
for i in range(5):
    idx = incorrect_indices[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Передбачено: {predicted_classes[idx]}\nРеально: {y_test[idx]}")
    plt.axis('off')
plt.suptitle("Неправильно розпізнані приклади (5)")
plt.tight_layout()
plt.show()
