import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Veri dizinleri
train_dir = '/Users/altun/Desktop/dataSet2/dataSet3/train'
test_dir = '/Users/altun/Desktop/dataSet2/dataSet3/test'

# Kayıt dizini oluşturma
save_dir = 'model_results/'
os.makedirs(save_dir, exist_ok=True)  # Klasör yoksa oluştur

# Veri genişletme için ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizasyon
    rotation_range=30,  # Görüntü döndürme
    width_shift_range=0.2,  # Yatay kaydırma
    height_shift_range=0.2,  # Dikey kaydırma
    shear_range=0.2,  # Kesme işlemi
    zoom_range=0.2,  # Yakınlaştırma
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest'  # Boşluk doldurma
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test setleri
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),  # Görüntü boyutunu ayarlayın
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),  # Görüntü boyutunu ayarlayın
    batch_size=32,
    class_mode='categorical'
)

# Modelin yapısı
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices),
          activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini yazdır
model.summary()

# EarlyStopping callback (overfitting'i engellemek için)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=20,  # Epoch sayısını arttırabilirsiniz
    validation_data=test_generator,
    callbacks=[early_stopping]  # EarlyStopping ekledim
)

# Modeli kaydetme
model_save_path = os.path.join(save_dir, 'saved_model.h5')
model.save(model_save_path)
print(f"Model '{model_save_path}' konumuna kaydedildi.")

# Eğitim ve test doğruluğu ve kaybını görselleştirme
plot_save_path = os.path.join(save_dir, 'training_plots.png')

plt.figure(figsize=(12, 6))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Doğruluk Eğrisi')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Kayıp Eğrisi')

# Grafikleri kaydetme
plt.tight_layout()
plt.savefig(plot_save_path)
print(f"Grafikler '{plot_save_path}' konumuna kaydedildi.")

# Grafikleri göster (isteğe bağlı)
plt.show()


