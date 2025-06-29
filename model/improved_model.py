import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, 
    BatchNormalization, Conv2D, MaxPooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CosineAnnealingScheduler, LearningRateScheduler
)
from tensorflow.keras.metrics import Precision, Recall, AUC
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility için seed ayarla
tf.random.set_seed(42)
np.random.seed(42)

class ImprovedSkinCancerCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['benign', 'melanoma', 'nevus']
        self.img_size = (224, 224)
        self.batch_size = 16  # Daha küçük batch size
        
    def create_transfer_learning_model(self):
        """Transfer learning ile gelişmiş model oluşturur."""
        print("🧠 Transfer Learning ile gelişmiş CNN modeli oluşturuluyor...")
        
        # Pre-trained EfficientNetB3 base model
        base_model = EfficientNetB3(
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
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')  # 3 sınıf
        ])
        
        # İlk eğitim - frozen base
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        self.model = model
        print("✅ Transfer learning modeli oluşturuldu!")
        print(f"📊 Toplam parametreler: {self.model.count_params():,}")
        
        return self.model
    
    def create_custom_cnn_model(self):
        """Geliştirilmiş özel CNN modeli oluşturur."""
        print("🧠 Gelişmiş özel CNN modeli oluşturuluyor...")
        
        self.model = Sequential([
            # İlk Conv Block
            Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # İkinci Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Üçüncü Conv Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Dördüncü Conv Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            
            # Fully Connected Layers
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        # Model derle
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
        
        print("✅ Gelişmiş özel CNN modeli oluşturuldu!")
        self.model.summary()
        
        return self.model
    
    def prepare_advanced_data(self):
        """Gelişmiş veri ön işleme ve augmentation."""
        print("\n📊 Gelişmiş veri hazırlanıyor...")
        
        # Agresif data augmentation - eğitim seti için
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Doğrulama için sadece normalizasyon
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Veri yükleme
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
        
        print("✅ Gelişmiş veri hazırlandı!")
        print(f"📈 Eğitim örnekleri: {train_generator.samples}")
        print(f"📊 Doğrulama örnekleri: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def calculate_class_weights(self, train_generator):
        """Dengesiz dataset için class weights hesaplar."""
        print("\n⚖️ Class weights hesaplanıyor...")
        
        # Sınıf etiketlerini al
        y_train = train_generator.classes
        
        # Class weights hesapla
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"📊 Class weights: {class_weight_dict}")
        return class_weight_dict
    
    def cosine_annealing(self, epoch, lr_max=0.001, lr_min=0.0001, cycle_length=10):
        """Cosine annealing learning rate scheduler."""
        cycle = np.floor(1 + epoch / cycle_length)
        x = np.abs(epoch / cycle_length - cycle)
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * x))
        return lr
    
    def train_advanced(self, epochs=100, use_transfer_learning=True):
        """Gelişmiş eğitim süreci."""
        print(f"\n🚀 Gelişmiş model eğitimi başlıyor... ({epochs} epoch)")
        
        # Model oluştur
        if use_transfer_learning:
            self.create_transfer_learning_model()
        else:
            self.create_custom_cnn_model()
        
        # Veri hazırla
        train_gen, val_gen = self.prepare_advanced_data()
        
        # Class weights hesapla
        class_weights = self.calculate_class_weights(train_gen)
        
        # Gelişmiş callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ModelCheckpoint(
                'model/best_improved_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=0.00001,
                verbose=1,
                cooldown=3
            ),
            LearningRateScheduler(
                lambda epoch: self.cosine_annealing(epoch),
                verbose=1
            )
        ]
        
        # İlk eğitim (frozen base)
        print("\n🔒 1. Aşama: Frozen base ile eğitim...")
        history1 = self.model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // self.batch_size),
            epochs=min(30, epochs//2),
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // self.batch_size),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Transfer learning varsa fine-tuning
        if use_transfer_learning:
            print("\n🔓 2. Aşama: Fine-tuning...")
            
            # Base model'i unfreeze et
            self.model.layers[0].trainable = True
            
            # Daha düşük learning rate ile yeniden compile
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall(), AUC()]
            )
            
            # Fine-tuning eğitimi
            history2 = self.model.fit(
                train_gen,
                steps_per_epoch=max(1, train_gen.samples // self.batch_size),
                epochs=epochs - min(30, epochs//2),
                validation_data=val_gen,
                validation_steps=max(1, val_gen.samples // self.batch_size),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                initial_epoch=len(history1.history['loss'])
            )
            
            # History'leri birleştir
            self.history = self.combine_histories(history1, history2)
        else:
            self.history = history1
        
        print("✅ Gelişmiş model eğitimi tamamlandı!")
        
        # Modeli kaydet
        self.save_improved_model()
        
        # Eğitim grafiklerini çiz
        self.plot_advanced_training_history()
        
        return self.history
    
    def combine_histories(self, hist1, hist2):
        """İki history'yi birleştirir."""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def save_improved_model(self):
        """Geliştirilmiş modeli kaydeder."""
        print("\n💾 Gelişmiş model kaydediliyor...")
        
        # Model klasörünü oluştur
        Path("model").mkdir(exist_ok=True)
        
        # Ana modeli güncelle
        self.model.save('model/skin_cancer_model.h5')
        
        # Backup olarak improved model
        self.model.save('model/improved_skin_cancer_model.h5')
        
        # Sınıf bilgileri
        class_info = {
            'class_names': self.class_names,
            'class_descriptions': {
                'benign': 'İyi huylu lezyon - Genellikle zararsız',
                'melanoma': 'Melanom - Kötü huylu, acil tıbbi müdahale gerekli',
                'nevus': 'Nevüs (Ben) - Genellikle zararsız pigment lezyonu'
            },
            'model_info': {
                'architecture': 'Transfer Learning + EfficientNetB3',
                'input_size': self.img_size,
                'total_params': int(self.model.count_params())
            }
        }
        
        with open('model/class_info.json', 'w', encoding='utf-8') as f:
            json.dump(class_info, f, ensure_ascii=False, indent=2)
        
        print("✅ Gelişmiş model kaydedildi!")
    
    def plot_advanced_training_history(self):
        """Gelişmiş eğitim grafiklerini çizer."""
        if self.history is None:
            print("❌ Eğitim geçmişi bulunamadı!")
            return
        
        print("\n📈 Gelişmiş eğitim grafikleri çiziliyor...")
        
        plt.switch_backend('Agg')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Eğitim Doğruluğu')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Doğrulama Doğruluğu')
        axes[0,0].set_title('Model Doğruluğu')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Doğruluk')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Eğitim Kaybı')
        axes[0,1].plot(self.history.history['val_loss'], label='Doğrulama Kaybı')
        axes[0,1].set_title('Model Kaybı')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Kayıp')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Precision
        axes[1,0].plot(self.history.history['precision'], label='Eğitim Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Doğrulama Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Recall
        axes[1,1].plot(self.history.history['recall'], label='Eğitim Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Doğrulama Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('model/improved_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Gelişmiş eğitim grafikleri kaydedildi!")
    
    def evaluate_improved_model(self):
        """Gelişmiş modeli değerlendirir."""
        print("\n🔍 Gelişmiş model test setinde değerlendiriliyor...")
        
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
        results = self.model.evaluate(test_generator, verbose=1)
        
        print(f"📊 Gelişmiş Test Sonuçları:")
        print(f"   - Test Kaybı: {results[0]:.4f}")
        print(f"   - Test Doğruluğu: {results[1]:.4f} ({results[1]*100:.2f}%)")
        print(f"   - Test Precision: {results[2]:.4f}")
        print(f"   - Test Recall: {results[3]:.4f}")
        print(f"   - Test AUC: {results[4]:.4f}")
        
        return results

def main():
    """Gelişmiş eğitim ana fonksiyonu."""
    print("🎯 SkinAI - Gelişmiş Cilt Kanseri Sınıflandırma Modeli")
    print("=" * 70)
    
    # Veri kontrolü
    data_path = Path("data/processed/train")
    if not data_path.exists():
        print("❌ Eğitim veri seti bulunamadı!")
        print("👉 Önce veri setini hazırlayın: python utils/download_data.py")
        return
    
    # Gelişmiş model oluştur ve eğit
    improved_cnn = ImprovedSkinCancerCNN()
    
    # Transfer learning ile eğitim (daha iyi sonuçlar için)
    history = improved_cnn.train_advanced(
        epochs=80, 
        use_transfer_learning=True
    )
    
    # Test değerlendirmesi
    improved_cnn.evaluate_improved_model()
    
    print("\n🎉 Gelişmiş model eğitimi tamamlandı!")
    print("👉 Web uygulamasını başlatın: python app.py")

if __name__ == "__main__":
    main() 