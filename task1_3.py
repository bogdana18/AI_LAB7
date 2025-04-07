import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Завантаження даних MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Нормалізація + reshape
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 3. One-hot encoding
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# 4. Вибір перших 40 зразків для навчання
x_train_subset = x_train[:40]
y_train_subset = y_train_cat[:40]

# 5. Побудова моделі: 2 приховані шари по 32 нейрони
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 6. Компіляція
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7. Навчання (batch=42, validation_split=0.15, epochs=7)
history = model.fit(
    x_train_subset, y_train_subset,
    batch_size=42,
    epochs=7,
    validation_split=0.15,
    verbose=2
)

# 8. Виведення перших 40 зразків
plt.figure(figsize=(10, 5))
for i in range(40):
    plt.subplot(4, 10, i + 1)
    plt.imshow(x_train_subset[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("40 використаних зразків")
plt.tight_layout()
plt.show()

# 9. Точність по епохах
for i, acc in enumerate(history.history['accuracy']):
    print(f"Епоха {i+1}: точність = {acc:.4f}, валідація = {history.history['val_accuracy'][i]:.4f}")

# 10. Перевірка на зразках 14, 20, 39
indices = [14, 20, 39]
samples = x_test[indices]
preds = model.predict(samples)
pred_labels = np.argmax(preds, axis=1)

plt.figure(figsize=(9, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Перевірка: {pred_labels[i]}, Результат: {y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.suptitle("Тестова перевірка: 14, 20, 39")
plt.show()

# 11. Аналіз 15 помилкових розпізнавань
all_preds = model.predict(x_test)
pred_classes = np.argmax(all_preds, axis=1)
incorrect = np.where(pred_classes != y_test)[0][:15]

plt.figure(figsize=(15, 4))
for i in range(15):
    idx = incorrect[i]
    plt.subplot(2, 8, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Перевірка: {pred_classes[idx]}\nРезультат: {y_test[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.suptitle("15 помилково розпізнаних зразків")
plt.show()

# 12. Збереження CSV результатів
df_results = pd.DataFrame({
    "True Label": y_test,
    "Predicted Label": pred_classes
})
df_results.to_csv("mnist_predictions_results.csv", index=False)
print("Результати збережено")
