import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Model dosyasının yolu
model_path = "model_results/saved_model.h5"  # Model dosyanızın tam yolu
model = load_model(model_path)

# Veri seti yolları
dataset_path = "/Users/altun/Desktop/dataSet2/dataSet3"
test_data_dir = os.path.join(dataset_path, "test")  # Test klasörünün yolu

# Sınıf isimlerini test klasörünün alt klasörlerinden al
class_names = sorted(os.listdir(test_data_dir))
print("Sınıf İsimleri:", class_names)

# Test veri setini hazırlayın
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Görselleri normalize et
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(100, 100),  # Modelin giriş boyutuna uygun olarak 100x100
    batch_size=32,
    class_mode="categorical",  # Çok sınıflı sınıflandırma
    shuffle=False  # Karışık olmadan sırayla işlesin
)




# Modelin tahminleri
y_pred_probs = model.predict(test_generator)  # Olasılık tahminleri
y_pred = np.argmax(y_pred_probs, axis=1)  # En yüksek olasılığa sahip sınıf
y_true = test_generator.classes  # Gerçek sınıf etiketleri

# Performans Analizi
conf_matrix = confusion_matrix(y_true, y_pred)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Görseli dosyaya kaydetme
output_path = "model_results/confusion_matrix.png"  # Kaydedilecek dosya yolu
plt.savefig(output_path)  # Resmi kaydet
print(f"Karışıklık matrisi {output_path} olarak kaydedildi.")

# Görselleştirmeyi ekranda göster

plt.show()
