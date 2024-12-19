import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Modeli yükle
model = load_model('model_results/saved_model.h5')

def get_class_names(folder_path):
    """
    Belirtilen klasörün altındaki klasörlerin isimlerini alır.
    :param folder_path: Ana klasörün yolu
    :return: Alt klasör isimlerinin bir listesi
    """
    class_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return class_names

# Test klasörünün yolunu belirtin
test_folder = "/Users/altun/Desktop/dataSet2/dataSet3/test"

# Fonksiyonu çağır ve isimleri class_name'e ata
class_names = get_class_names(test_folder)
class_labels = class_names  # Sınıf etiketlerini tanımla

print("Class Names:", class_names)

# Web kamerasını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı kullanır

if not cap.isOpened():
    print("Web kamerası açılamadı!")
    exit()

print("Tahmin için 'q' tuşuna basarak çıkabilirsiniz.")

while True:
    # Kameradan bir kare yakala
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı, çıkılıyor...")
        break

    # Görüntüyü ön işleme
    input_image = cv2.resize(frame, (100, 100))  # Modelinize uygun boyut
    input_image = img_to_array(input_image)  # Görüntüyü array'e dönüştür
    input_image = np.expand_dims(input_image, axis=0)  # Batch boyutu ekle
    input_image = input_image / 255.0  # Normalizasyon

    # Modelle tahmin yap
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # En olası sınıfın indeksini al
    confidence = predictions[0][predicted_class]  # Tahmin edilen sınıfa olan güven
    predicted_label = class_labels[predicted_class]

    # Sonucu görüntüde göster
    cv2.putText(frame, f"Tahmin: {predicted_label} ({confidence * 100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Kameradan alınan görüntüyü göster
    cv2.imshow("Web Kamera - Model Tahmini", frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()