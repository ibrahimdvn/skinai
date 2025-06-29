import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, 
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class EnhancedSkinCancerCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['benign', 'melanoma', 'nevus']
        self.img_size = (224, 224)
        self.batch_size = 8  # Küçük dataset için daha küçük batch
        
    def create_enhanced_model(self):
        """Transfer learning ile gelişmiş model."""
        print("🧠 Transfer Learning modeli oluşturuluyor...")
        
        # Pre-trained EfficientNetB0 (daha hafif ama etkili)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # İlk katmanları dondur
        base_model.trainable = False
        
        # Model mimarisi
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        self.model = model
        print("✅ Enhanced model hazır!")
        print(f"📊 Toplam parametreler: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_enhanced_data(self):
        """Gelişmiş data augmentation."""
        print("\n📊 Enhanced data hazırlanıyor...")
        
        # Agresif augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            'data/processed/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'data/processed/validation',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        print(f"📈 Eğitim örnekleri: {train_generator.samples}")
        print(f"📊 Doğrulama örnekleri: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def calculate_class_weights(self, train_generator):
        """Class weights hesaplar."""
        y_train = train_generator.classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"📊 Class weights: {class_weight_dict}")
        return class_weight_dict
    
    def train_enhanced(self, epochs=100):
        """Gelişmiş eğitim."""
        print(f"\n🚀 Enhanced eğitim başlıyor... ({epochs} epoch)")
        
        # Model oluştur
        self.create_enhanced_model()
        
        # Veri hazırla
        train_gen, val_gen = self.prepare_enhanced_data()
        
        # Class weights
        class_weights = self.calculate_class_weights(train_gen)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ModelCheckpoint(
                'model/enhanced_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Aşama 1: Frozen base
        print("\n🔒 Aşama 1: Frozen base eğitimi...")
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // self.batch_size),
            epochs=min(40, epochs//2),
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // self.batch_size),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Aşama 2: Fine-tuning
        print("\n🔓 Aşama 2: Fine-tuning...")
        
        # Base model unfreeze
        self.model.layers[0].trainable = True
        
        # Düşük learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        # Fine-tuning eğitimi
        history2 = self.model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // self.batch_size),
            epochs=epochs - min(40, epochs//2),
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // self.batch_size),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            initial_epoch=len(self.history.history['loss'])
        )
        
        print("✅ Enhanced eğitim tamamlandı!")
        
        # Ana modeli güncelle
        self.model.save('model/skin_cancer_model.h5')
        self.model.save('model/best_model.h5')
        
        self.plot_training()
        
        return self.history
    
    def plot_training(self):
        """Eğitim grafiklerini çizer."""
        if self.history is None:
            return
        
        plt.switch_backend('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Eğitim')
        ax1.plot(self.history.history['val_accuracy'], label='Doğrulama')
        ax1.set_title('Model Doğruluğu')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Doğruluk')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Eğitim')
        ax2.plot(self.history.history['val_loss'], label='Doğrulama')
        ax2.set_title('Model Kaybı')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Kayıp')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('model/enhanced_training_history.png', dpi=300)
        plt.close()
        
        print("✅ Enhanced grafikler kaydedildi!")
    
    def evaluate_enhanced(self):
        """Test değerlendirmesi."""
        print("\n🔍 Enhanced model test ediliyor...")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'data/processed/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        results = self.model.evaluate(test_generator, verbose=1)
        
        print(f"📊 Enhanced Test Sonuçları:")
        print(f"   - Test Kaybı: {results[0]:.4f}")
        print(f"   - Test Doğruluğu: {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"   - Test Precision: {results[2]:.4f}")
        print(f"   - Test Recall: {results[3]:.4f}")
        
        return results

def main():
    """Ana fonksiyon."""
    print("🎯 SkinAI - Enhanced Model Training")
    print("=" * 50)
    
    # Veri kontrolü
    data_path = Path("data/processed/train")
    if not data_path.exists():
        print("❌ Veri bulunamadı!")
        print("👉 Önce: python utils/download_data.py")
        return
    
    # Enhanced model
    enhanced_cnn = EnhancedSkinCancerCNN()
    
    # Eğitim
    history = enhanced_cnn.train_enhanced(epochs=80)
    
    # Test
    enhanced_cnn.evaluate_enhanced()
    
    print("\n🎉 Enhanced model hazır!")
    print("👉 Web uygulaması: python app.py")

if __name__ == "__main__":
    main() 