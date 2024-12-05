import os
import shutil
from sklearn.model_selection import train_test_split

# Veri seti ana klasör yolu
dataset_path = "/Users/altun/Desktop/dataSet2/dataSet3"  # Kendi dataset yolunu belirt
train_path = os.path.join(dataset_path, "train")  # train klasörü dataset içinde
test_path = os.path.join(dataset_path, "test")  # test klasörü dataset içinde

# Train-test oranını belirle
train_ratio = 0.8

# Train ve test klasörlerini oluştur
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Ana klasördeki her bir alt klasörü oku
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):  # Eğer klasör değilse atla
        continue

    # Kategori altındaki tüm dosyaları listele
    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    # Eğer dosya yoksa kategoriyi atla
    if len(files) == 0:
        print(f"{category} kategorisinde dosya yok, atlanıyor.")
        continue

    # Train ve test için kategori klasörlerini oluştur
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

    # Dosyaları train ve test olarak böl
    train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=42)

    # Train dosyalarını taşı
    for file in train_files:
        shutil.move(os.path.join(category_path, file), os.path.join(train_path, category, file))

    # Test dosyalarını taşı
    for file in test_files:
        shutil.move(os.path.join(category_path, file), os.path.join(test_path, category, file))

print("Veri seti dataset içinde train ve test olarak ayrıldı!")