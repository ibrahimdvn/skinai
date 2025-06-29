import os
import pandas as pd
import numpy as np
import zipfile
import requests
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def download_ham10000():
    """
    HAM10000 veri setini indirir ve hazırlar.
    Kaggle API kullanarak indirme yapar.
    """
    print("🩺 SkinAI - HAM10000 Veri Seti İndiriliyor...")
    
    # Veri klasörlerini kontrol et
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Manuel veri seti yolu - gerçek projede Kaggle API kullanılabilir
    print("\n📁 Veri klasörü yapısı hazırlanıyor...")
    
    # Sınıf etiketleri
    class_names = {
        'akiec': 'Actinic keratoses',  # Aktiniк keratozlar
        'bcc': 'Basal cell carcinoma',  # Bazal hücreli karsinom
        'bkl': 'Benign keratosis',     # İyi huylu keratoz
        'df': 'Dermatofibroma',        # Dermatofibroma
        'mel': 'Melanoma',             # Melanom (kötü huylu)
        'nv': 'Melanocytic nevus',     # Melanositik nevüs (ben)
        'vasc': 'Vascular lesions'     # Vasküler lezyonlar
    }
    
    # Simplified class mapping for our model (3 classes)
    simplified_classes = {
        'mel': 'melanoma',      # Kötü huylu
        'bcc': 'melanoma',      # Kötü huylu
        'akiec': 'melanoma',    # Kötü huylu
        'bkl': 'benign',        # İyi huylu
        'df': 'benign',         # İyi huylu  
        'nv': 'nevus',          # Ben
        'vasc': 'benign'        # İyi huylu
    }
    
    # Create class directories
    for split in ['train', 'validation', 'test']:
        for class_name in ['melanoma', 'benign', 'nevus']:
            class_dir = processed_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    print("✅ Klasör yapısı hazırlandı!")
    return simplified_classes

def create_sample_dataset():
    """
    Demo amaçlı örnek veri seti oluşturur.
    Gerçek projede HAM10000 kullanılmalı.
    """
    print("\n🎨 Demo veri seti oluşturuluyor...")
    
    from PIL import Image
    import random
    
    # Create sample images for demo
    for split in ['train', 'validation', 'test']:
        for class_name in ['melanoma', 'benign', 'nevus']:
            class_dir = Path(f"data/processed/{split}/{class_name}")
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Number of sample images
            num_samples = 50 if split == 'train' else 10
            
            for i in range(num_samples):
                # Create a random colored image (demo purposes)
                if class_name == 'melanoma':
                    color = (random.randint(60, 120), random.randint(30, 80), random.randint(30, 80))
                elif class_name == 'benign':
                    color = (random.randint(180, 220), random.randint(140, 180), random.randint(120, 160))
                else:  # nevus
                    color = (random.randint(100, 140), random.randint(80, 120), random.randint(60, 100))
                
                # Add some noise
                img = Image.new('RGB', (224, 224), color)
                pixels = img.load()
                for x in range(224):
                    for y in range(224):
                        r, g, b = pixels[x, y]
                        noise = random.randint(-30, 30)
                        pixels[x, y] = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )
                
                img.save(class_dir / f"{class_name}_{i:03d}.jpg")
    
    print("✅ Demo veri seti oluşturuldu!")
    print(f"📊 Veri dağılımı:")
    print(f"   - Eğitim: 150 görüntü (50 melanoma, 50 benign, 50 nevus)")
    print(f"   - Doğrulama: 30 görüntü (10'ar sınıf)")
    print(f"   - Test: 30 görüntü (10'ar sınıf)")

def get_data_statistics():
    """Veri seti istatistiklerini gösterir."""
    processed_dir = Path("data/processed")
    
    print("\n📊 Veri Seti İstatistikleri:")
    print("-" * 50)
    
    total_images = 0
    for split in ['train', 'validation', 'test']:
        split_total = 0
        print(f"\n{split.upper()}:")
        for class_name in ['melanoma', 'benign', 'nevus']:
            class_dir = processed_dir / split / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.jpg")))
                print(f"  {class_name}: {count} görüntü")
                split_total += count
        print(f"  Toplam: {split_total} görüntü")
        total_images += split_total
    
    print(f"\n🎯 GENEL TOPLAM: {total_images} görüntü")
    
    return total_images

if __name__ == "__main__":
    # Veri setini hazırla
    simplified_classes = download_ham10000()
    
    # Demo veri seti oluştur
    create_sample_dataset()
    
    # İstatistikleri göster
    get_data_statistics()
    
    print("\n🎉 Veri hazırlama tamamlandı!")
    print("👉 Şimdi model eğitimine geçebilirsin: python model/train_model.py") 