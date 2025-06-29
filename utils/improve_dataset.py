import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

class DatasetImprover:
    def __init__(self):
        self.real_melanoma_dir = "Melanom görüntüleri"
        self.processed_dir = "data/processed"
        self.target_size = (224, 224)
        
    def copy_real_melanoma_images(self):
        """Gerçek melanom görüntülerini dataset'e ekler."""
        print("🖼️ Gerçek melanom görüntüleri kopyalanıyor...")
        
        real_images = list(Path(self.real_melanoma_dir).glob("*.jpg"))
        train_melanoma_dir = Path(f"{self.processed_dir}/train/melanoma")
        
        for i, img_path in enumerate(real_images):
            # Train setine ekle
            dest_path = train_melanoma_dir / f"real_melanoma_{i:03d}.jpg"
            shutil.copy2(img_path, dest_path)
            print(f"   ✅ {img_path.name} -> {dest_path.name}")
            
        print(f"📈 {len(real_images)} gerçek melanom görüntüsü eklendi!")
        
    def generate_augmented_images(self, source_dir, target_dir, num_augmentations=20):
        """Aggressive data augmentation."""
        print(f"🔄 {source_dir} için data augmentation...")
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Advanced augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.6, 1.4],
            channel_shift_range=0.3,
            fill_mode='nearest'
        )
        
        source_images = list(Path(source_dir).glob("*.jpg"))
        
        for img_path in source_images:
            # Load and prepare image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Generate augmented images
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                if i >= num_augmentations:
                    break
                
                # Convert back to image
                aug_img = (batch[0] * 255).astype(np.uint8)
                aug_img = Image.fromarray(aug_img)
                
                # Save augmented image
                base_name = img_path.stem
                aug_path = Path(target_dir) / f"{base_name}_aug_{i:03d}.jpg"
                aug_img.save(aug_path, quality=95)
                
                i += 1
        
        print(f"   ✅ {len(source_images) * num_augmentations} augmented görüntü oluşturuldu!")
    
    def apply_advanced_preprocessing(self, image_path):
        """Gelişmiş görüntü ön işleme."""
        img = cv2.imread(str(image_path))
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Edge enhancement
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def enhance_existing_dataset(self):
        """Mevcut dataset'i iyileştirir."""
        print("🚀 Dataset iyileştirme başlıyor...")
        
        # 1. Gerçek melanom görüntülerini ekle
        self.copy_real_melanoma_images()
        
        # 2. Her sınıf için aggressive augmentation
        classes = ['benign', 'melanoma', 'nevus']
        
        for class_name in classes:
            source_dir = f"{self.processed_dir}/train/{class_name}"
            
            # Mevcut görüntü sayısını kontrol et
            existing_images = list(Path(source_dir).glob("*.jpg"))
            current_count = len(existing_images)
            
            print(f"\n📊 {class_name.upper()}: {current_count} mevcut görüntü")
            
            # Her görüntü için 30 augmentation (daha agresif)
            self.generate_augmented_images(
                source_dir, 
                source_dir, 
                num_augmentations=30
            )
            
            # Yeni görüntü sayısı
            new_images = list(Path(source_dir).glob("*.jpg"))
            new_count = len(new_images)
            
            print(f"   📈 Yeni toplam: {new_count} görüntü (+{new_count - current_count})")
        
        # 3. Validation ve test setlerini de genişlet
        self.expand_validation_test_sets()
        
    def expand_validation_test_sets(self):
        """Validation ve test setlerini genişletir."""
        print("\n📊 Validation ve test setleri genişletiliyor...")
        
        classes = ['benign', 'melanoma', 'nevus']
        
        for class_name in classes:
            # Validation set
            val_dir = f"{self.processed_dir}/validation/{class_name}"
            self.generate_augmented_images(val_dir, val_dir, num_augmentations=15)
            
            # Test set
            test_dir = f"{self.processed_dir}/test/{class_name}"
            self.generate_augmented_images(test_dir, test_dir, num_augmentations=15)
    
    def create_balanced_dataset(self):
        """Dengeli bir dataset oluşturur."""
        print("\n⚖️ Dataset dengeleme...")
        
        classes = ['benign', 'melanoma', 'nevus']
        
        # Her sınıfın görüntü sayısını say
        class_counts = {}
        for class_name in classes:
            train_dir = f"{self.processed_dir}/train/{class_name}"
            images = list(Path(train_dir).glob("*.jpg"))
            class_counts[class_name] = len(images)
            print(f"   {class_name}: {len(images)} görüntü")
        
        # En fazla görüntüye sahip sınıfı bul
        max_count = max(class_counts.values())
        print(f"\n🎯 Hedef: Her sınıf için {max_count} görüntü")
        
        # Diğer sınıfları bu sayıya çıkar
        for class_name in classes:
            current_count = class_counts[class_name]
            needed = max_count - current_count
            
            if needed > 0:
                print(f"   {class_name} için {needed} ek görüntü oluşturuluyor...")
                
                source_dir = f"{self.processed_dir}/train/{class_name}"
                self.generate_augmented_images(
                    source_dir, 
                    source_dir, 
                    num_augmentations=needed // current_count + 1
                )
    
    def show_dataset_stats(self):
        """Dataset istatistiklerini gösterir."""
        print("\n📊 DATASET İSTATİSTİKLERİ")
        print("=" * 50)
        
        sets = ['train', 'validation', 'test']
        classes = ['benign', 'melanoma', 'nevus']
        
        total_images = 0
        
        for set_name in sets:
            print(f"\n{set_name.upper()} SET:")
            set_total = 0
            
            for class_name in classes:
                class_dir = f"{self.processed_dir}/{set_name}/{class_name}"
                images = list(Path(class_dir).glob("*.jpg"))
                count = len(images)
                set_total += count
                total_images += count
                
                print(f"   {class_name:8}: {count:4} görüntü")
            
            print(f"   Toplam  : {set_total:4} görüntü")
        
        print(f"\n🎯 GENEL TOPLAM: {total_images} görüntü")
        print("=" * 50)

def main():
    """Ana fonksiyon."""
    print("🎯 SkinAI Dataset İyileştirme")
    print("=" * 40)
    
    improver = DatasetImprover()
    
    # Mevcut durumu göster
    print("\n📊 Mevcut dataset durumu:")
    improver.show_dataset_stats()
    
    # Dataset'i iyileştir
    improver.enhance_existing_dataset()
    
    # Balanced dataset oluştur
    improver.create_balanced_dataset()
    
    # Final durumu göster
    print("\n🎉 İyileştirme tamamlandı!")
    improver.show_dataset_stats()
    
    print("\n👉 Şimdi enhanced model'i yeniden eğitin:")
    print("   python model/enhanced_model.py")

if __name__ == "__main__":
    main() 