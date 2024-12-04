from tensorflow.keras.models import load_model

# Modeli yükleme
model = load_model('model_results/saved_model.h5')

# Modelin özetini yazdırarak doğruluğunu kontrol edebilirsiniz
model.summary()
import numpy as np
from tensorflow.keras.preprocessing import image

# Yeni bir görüntü dosyasını yükleme ve ön işleme
img_path = 'Apple Braeburn_10.jpg'

img = image.load_img(img_path, target_size=(100, 100))  # Görüntü boyutunu 100x100 yapıyoruz
img_array = image.img_to_array(img)  # Görüntüyü array formatına çevirme
img_array = np.expand_dims(img_array, axis=0)  # Tek bir görüntü olduğunda batch boyutunu ekliyoruz
img_array /= 255.0  # Modelinizi eğitirken kullandığınız normalizasyon

# Tahmin yapma
predictions = model.predict(img_array)

# Tahmin sonucunu yazdırma
predicted_class = np.argmax(predictions, axis=1)  # Tahmin edilen sınıfın indeksini alır
print("Tahmin Edilen Sınıf:", predicted_class)