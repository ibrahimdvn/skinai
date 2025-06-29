import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Reproducibility için seed ayarla
tf.random.set_seed(42)
np.random.seed(42)

class SkinCancerCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['benign', 'melanoma', 'nevus']
        self.img_size = (224, 224)
        self.batch_size = 32
        
    def create_model(self):
        """CNN modelini oluşturur."""
        print("🧠 CNN Modeli oluşturuluyor...")
        
        self.model = Sequential([
            # İlk Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # İkinci Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Üçüncü Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Dördüncü Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')  # 3 sınıf: benign, melanoma, nevus
        ])
        
        # Model derle
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model oluşturuldu!")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self):
        """Veri ön işleme ve data generator hazırlama."""
        print("\n📊 Veri hazırlanıyor...")
        
        # Veri artırma (Data Augmentation) - sadece eğitim seti için
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Doğrulama ve test setleri için sadece normalizasyon
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Veri yükleme
        train_generator = train_datagen.flow_from_directory(
            'data/processed/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'data/processed/validation',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        print("✅ Veri hazırlandı!")
        print(f"📈 Eğitim örnekleri: {train_generator.samples}")
        print(f"📊 Doğrulama örnekleri: {validation_generator.samples}")
        print(f"🏷️  Sınıf etiketleri: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def train(self, epochs=50):
        """Modeli eğitir."""
        print(f"\n🚀 Model eğitimi başlıyor... ({epochs} epoch)")
        
        # Veri hazırla
        train_gen, val_gen = self.prepare_data()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'model/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Model eğitimi
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // self.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model eğitimi tamamlandı!")
        
        # Modeli kaydet
        self.save_model()
        
        # Eğitim grafiklerini çiz
        self.plot_training_history()
        
        return self.history
    
    def save_model(self):
        """Eğitilmiş modeli kaydeder."""
        print("\n💾 Model kaydediliyor...")
        
        # Model klasörünü oluştur
        Path("model").mkdir(exist_ok=True)
        
        # Modeli kaydet
        self.model.save('model/skin_cancer_model.h5')
        
        # Sınıf isimlerini kaydet
        class_info = {
            'class_names': self.class_names,
            'class_descriptions': {
                'benign': 'İyi huylu lezyon - Genellikle zararsız',
                'melanoma': 'Melanom - Kötü huylu, acil tıbbi müdahale gerekli',
                'nevus': 'Nevüs (Ben) - Genellikle zararsız pigment lezyonu'
            }
        }
        
        with open('model/class_info.json', 'w', encoding='utf-8') as f:
            json.dump(class_info, f, ensure_ascii=False, indent=2)
        
        print("✅ Model kaydedildi: model/skin_cancer_model.h5")
        print("✅ Sınıf bilgileri kaydedildi: model/class_info.json")
    
    def plot_training_history(self):
        """Eğitim geçmişini görselleştirir."""
        if self.history is None:
            print("❌ Eğitim geçmişi bulunamadı!")
            return
        
        print("\n📈 Eğitim grafikleri çiziliyor...")
        
        # Matplotlib backend ayarla
        plt.switch_backend('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy grafiği
        ax1.plot(self.history.history['accuracy'], label='Eğitim Doğruluğu')
        ax1.plot(self.history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        ax1.set_title('Model Doğruluğu')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Doğruluk')
        ax1.legend()
        ax1.grid(True)
        
        # Loss grafiği
        ax2.plot(self.history.history['loss'], label='Eğitim Kaybı')
        ax2.plot(self.history.history['val_loss'], label='Doğrulama Kaybı')
        ax2.set_title('Model Kaybı')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Kayıp')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Eğitim grafikleri kaydedildi: model/training_history.png")
    
    def evaluate_model(self):
        """Modeli test setinde değerlendirir."""
        print("\n🔍 Model test setinde değerlendiriliyor...")
        
        # Test veri generatörü
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'data/processed/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        # Değerlendirme
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        print(f"📊 Test Sonuçları:")
        print(f"   - Test Kaybı: {test_loss:.4f}")
        print(f"   - Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy

def main():
    """Ana eğitim fonksiyonu."""
    print("🎯 SkinAI - Cilt Kanseri Sınıflandırma Modeli Eğitimi")
    print("=" * 60)
    
    # Veri kontrolü
    data_path = Path("data/processed/train")
    if not data_path.exists():
        print("❌ Eğitim veri seti bulunamadı!")
        print("👉 Önce veri setini hazırlayın: python utils/download_data.py")
        return
    
    # Model oluştur ve eğit
    skin_cnn = SkinCancerCNN()
    model = skin_cnn.create_model()
    
    # Eğitim
    history = skin_cnn.train(epochs=30)  # Demo için 30 epoch
    
    # Test değerlendirmesi
    skin_cnn.evaluate_model()
    
    print("\n🎉 Eğitim tamamlandı!")
    print("👉 Web uygulamasını başlatın: python app.py")

if __name__ == "__main__":
    main() 