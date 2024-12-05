import os

def delete_empty_folders(folder_path):
    # Klasörün içindeki tüm dosya ve alt klasörleri dolaş
    for root, dirs, files in os.walk(folder_path, topdown=False):  # `topdown=False` alt klasörleri önce kontrol eder
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Eğer klasör boşsa sil
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Boş klasör silindi: {dir_path}")

# Kullanım
dataset_path = "/Users/altun/Desktop/dataSet2/dataSet3"  # Dataset yolunu belirt
delete_empty_folders(dataset_path)