#!/usr/bin/env python3
"""
🎯 SkinAI - HAM10000 Gelişmiş Model Eğitimi
================================================
HAM10000 dataset'i kullanarak güçlü bir cilt kanseri tespit modeli eğitir.

Dataset Mapping:
- mel (melanoma) → melanoma
- nv (nevus) → nevus  
- bkl, bcc, akiec, df → benign

Author: SkinAI Team
Date: 2025-06-28
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import json

# Konfigürasyon
CONFIG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EPOCHS': 40,
    'LEARNING_RATE': 0.001,
    'VALIDATION_SPLIT': 0.2,
    'TEST_SPLIT': 0.1,
    'NUM_CLASSES': 3,
    'MODEL_NAME': 'ham10k_model.h5'
}

class HAM10kTrainer:
    def __init__(self):
        self.metadata_path = 'model/archive/HAM10000_metadata.csv'
        self.images_part1 = 'model/archive/HAM10000_images_part_1'
        self.images_part2 = 'model/archive/HAM10000_images_part_2'
        self.output_dir = 'data/ham10k_processed'
        
        # Sınıf mapping: HAM10000 -> Bizim sistem
        self.class_mapping = {
            'mel': 'melanoma',      # Melanoma
            'nv': 'nevus',          # Nevus (mole)
            'bkl': 'benign',        # Benign keratosis
            'bcc': 'benign',        # Basal cell carcinoma  
            'akiec': 'benign',      # Actinic keratoses
            'df': 'benign',         # Dermatofibroma
            'vasc': 'benign'        # Vascular lesion
        }
        
        self.class_names = ['benign', 'melanoma', 'nevus']
        
    def load_and_prepare_data(self):
        """HAM10000 metadata'sını yükler ve veriyi hazırlar"""
        print("📊 HAM10000 metadata yükleniyor...")
        
        df = pd.read_csv(self.metadata_path)
        print(f"✅ {len(df)} görüntü metadata'sı yüklendi")
        
        print("\n🔍 Orijinal sınıf dağılımı:")
        print(df['dx'].value_counts())
        
        # Sınıfları mapping yap
        df['mapped_class'] = df['dx'].map(self.class_mapping)
        
        print("\n🎯 Yeni sınıf dağılımı:")
        print(df['mapped_class'].value_counts())
        
        # Görüntü dosyalarının varlığını kontrol et
        valid_images = []
        for idx, row in df.iterrows():
            image_id = row['image_id']
            
            img_path1 = os.path.join(self.images_part1, f"{image_id}.jpg")
            img_path2 = os.path.join(self.images_part2, f"{image_id}.jpg")
            
            if os.path.exists(img_path1):
                valid_images.append({
                    'image_id': image_id,
                    'image_path': img_path1,
                    'class': row['mapped_class']
                })
            elif os.path.exists(img_path2):
                valid_images.append({
                    'image_id': image_id,
                    'image_path': img_path2,
                    'class': row['mapped_class']
                })
        
        print(f"✅ {len(valid_images)} geçerli görüntü bulundu")
        self.df = pd.DataFrame(valid_images)
        return self.df
    
    def organize_dataset(self):
        """Dataset'i train/val/test olarak organize eder"""
        print("\n📁 Dataset organize ediliyor...")
        
        # Output dizinlerini oluştur
        for split in ['train', 'validation', 'test']:
            for class_name in self.class_names:
                os.makedirs(os.path.join(self.output_dir, split, class_name), exist_ok=True)
        
        # Stratified split
        train_data, temp_data = train_test_split(
            self.df, 
            test_size=CONFIG['VALIDATION_SPLIT'] + CONFIG['TEST_SPLIT'],
            stratify=self.df['class'],
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=CONFIG['TEST_SPLIT'] / (CONFIG['VALIDATION_SPLIT'] + CONFIG['TEST_SPLIT']),
            stratify=temp_data['class'],
            random_state=42
        )
        
        # Görüntüleri kopyala ve resize et
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()} seti organize ediliyor...")
            
            for idx, row in split_df.iterrows():
                src_path = row['image_path']
                dst_path = os.path.join(
                    self.output_dir, 
                    split_name, 
                    row['class'], 
                    f"{row['image_id']}.jpg"
                )
                
                try:
                    img = Image.open(src_path)
                    img = img.convert('RGB')
                    img = img.resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
                    img.save(dst_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"⚠️  Hata: {src_path} - {e}")
            
            class_counts = split_df['class'].value_counts()
            print(f"📊 {split_name} - Toplam: {len(split_df)}")
            for class_name, count in class_counts.items():
                print(f"   {class_name}: {count}")
        
        print("✅ Dataset organizasyonu tamamlandı!")
        return train_data, val_data, test_data
    
    def create_data_generators(self):
        """Gelişmiş data augmentation ile veri generatörlerini oluşturur"""
        print("\n🔄 Data generators oluşturuluyor...")
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        # Validation/Test
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.output_dir, 'train'),
            target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.output_dir, 'validation'),
            target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_datagen.flow_from_directory(
            os.path.join(self.output_dir, 'test'),
            target_size=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=False
        )
        
        # Class weights hesapla
        class_counts = self.df['class'].value_counts()
        total_samples = len(self.df)
        
        class_weights = {}
        for i, class_name in enumerate(self.class_names):
            class_weights[i] = total_samples / (len(self.class_names) * class_counts[class_name])
        
        print(f"🎯 Class weights: {class_weights}")
        
        return train_generator, validation_generator, test_generator, class_weights
    
    def create_model(self):
        """EfficientNetB0 tabanlı model oluşturur"""
        print("\n🧠 Model oluşturuluyor...")
        
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3)
        )
        
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(CONFIG['NUM_CLASSES'], activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=x)
        
        model.compile(
            optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model oluşturuldu: {model.count_params():,} parametre")
        return model
    
    def train_model(self, model, train_gen, val_gen, class_weights):
        """Modeli eğitir"""
        print("\n🚀 Model eğitimi başlıyor...")
        
        callbacks = [
            ModelCheckpoint(
                f'model/{CONFIG["MODEL_NAME"]}',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Phase 1: Transfer Learning
        print("📚 Phase 1: Transfer Learning")
        history1 = model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print("\n🔧 Phase 2: Fine-tuning")
        model.layers[0].trainable = True
        
        model.compile(
            optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            epochs=CONFIG['EPOCHS'] - 15,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # History birleştir
        history = {}
        for key in history1.history.keys():
            history[key] = history1.history[key] + history2.history[key]
        
        return model, history
    
    def evaluate_model(self, model, test_gen):
        """Modeli test eder"""
        print("\n📊 Model değerlendiriliyor...")
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=1)
        
        print(f"\n🎯 Test Sonuçları:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc
    
    def save_results(self, history, test_results):
        """Sonuçları kaydeder"""
        print("\n💾 Sonuçlar kaydediliyor...")
        
        # Model bilgileri
        model_info = {
            'model_name': CONFIG['MODEL_NAME'],
            'dataset': 'HAM10000',
            'classes': self.class_names,
            'total_samples': len(self.df),
            'class_distribution': self.df['class'].value_counts().to_dict(),
            'test_accuracy': float(test_results[1]),
            'test_loss': float(test_results[0])
        }
        
        with open('model/ham10k_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Training plots
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('model/ham10k_training_history.png', dpi=300)
        plt.close()
        
        print("✅ Sonuçlar kaydedildi!")
    
    def run_training(self):
        """Tam eğitim pipeline'ını çalıştırır"""
        print("🎯 HAM10000 Gelişmiş Model Eğitimi")
        print("=" * 50)
        
        try:
            # Pipeline adımları
            self.load_and_prepare_data()
            train_df, val_df, test_df = self.organize_dataset()
            train_gen, val_gen, test_gen, class_weights = self.create_data_generators()
            model = self.create_model()
            trained_model, history = self.train_model(model, train_gen, val_gen, class_weights)
            test_results = self.evaluate_model(trained_model, test_gen)
            self.save_results(history, test_results)
            
            print("\n🎉 HAM10000 model eğitimi tamamlandı!")
            print(f"💾 Model kaydedildi: model/{CONFIG['MODEL_NAME']}")
            print(f"📊 Test Accuracy: {test_results[1]:.4f}")
            
            return trained_model, history, test_results
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")
            raise e

if __name__ == "__main__":
    # GPU ayarları
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU kullanılıyor: {len(gpus)} GPU")
        except RuntimeError as e:
            print(f"⚠️  GPU ayarı hatası: {e}")
    else:
        print("💻 CPU kullanılıyor")
    
    # Training başlat
    trainer = HAM10kTrainer()
    model, history, results = trainer.run_training() 