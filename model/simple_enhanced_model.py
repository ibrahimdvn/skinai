import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class SimpleEnhancedModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['benign', 'melanoma', 'nevus']
        self.img_size = (224, 224)
        self.batch_size = 16
        
    def create_working_model(self):
        """Basit ama çalışan model oluşturur."""
        print("🧠 Basit Enhanced Model oluşturuluyor...")
        
        # MobileNetV2 base model (daha stabil)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # İlk katmanları dondur
        base_model.trainable = False
        
        # Basit model mimarisi
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Basit Enhanced Model hazır!")
        print(f"📊 Toplam parametreler: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_simple_data(self):
        """Basit veri hazırlama."""
        print("\n📊 Veri hazırlanıyor...")
        
        # Basit augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
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
    
    def train_simple(self, epochs=30):
        """Basit eğitim süreci."""
        print(f"\n🚀 Basit eğitim başlıyor... ({epochs} epoch)")
        
        # Model oluştur
        self.create_working_model()
        
        # Veri hazırla
        train_gen, val_gen = self.prepare_simple_data()
        
        # Basit callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'model/simple_enhanced.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Eğitim
        print("\n🔄 Model eğitimi...")
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=min(50, train_gen.samples // self.batch_size),  # Daha az step
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=min(10, val_gen.samples // self.batch_size),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Basit eğitim tamamlandı!")
        
        # Ana modeli güncelle
        self.model.save('model/skin_cancer_model.h5')
        self.model.save('model/best_model.h5')
        
        # Test
        self.test_model()
        
        return self.history
    
    def test_model(self):
        """Modeli hızlı test eder."""
        print("\n🧪 Model test ediliyor...")
        
        # Random test
        random_input = np.random.random((1, 224, 224, 3))
        prediction = self.model.predict(random_input, verbose=0)[0]
        
        print("Random input sonuçları:")
        for i, class_name in enumerate(self.class_names):
            confidence = prediction[i] * 100
            print(f"   {class_name:8}: {confidence:5.1f}%")
        
        # Farklı sonuçlar alıyor muyuz test et
        predictions = []
        for _ in range(3):
            random_input = np.random.random((1, 224, 224, 3))
            pred = self.model.predict(random_input, verbose=0)[0]
            predictions.append(pred)
        
        # Tahminler farklı mı?
        all_same = all(np.allclose(predictions[0], pred, atol=0.01) for pred in predictions)
        
        if all_same:
            print("⚠️  Model hala aynı sonuçlar veriyor!")
        else:
            print("✅ Model farklı sonuçlar veriyor - Başarılı!")
    
    def save_working_model(self):
        """Çalışan modeli kaydet."""
        print("\n💾 Çalışan model kaydediliyor...")
        
        # Model klasörünü oluştur
        Path("model").mkdir(exist_ok=True)
        
        # Ana modeli güncelle
        self.model.save('model/skin_cancer_model.h5')
        
        # Sınıf bilgileri
        class_info = {
            'class_names': self.class_names,
            'class_descriptions': {
                'benign': 'İyi huylu lezyon - Genellikle zararsız',
                'melanoma': 'Melanom - Kötü huylu, acil tıbbi müdahale gerekli',
                'nevus': 'Nevüs (Ben) - Genellikle zararsız pigment lezyonu'
            },
            'model_info': {
                'architecture': 'MobileNetV2 + Transfer Learning',
                'input_size': self.img_size,
                'total_params': int(self.model.count_params()),
                'working': True
            }
        }
        
        with open('model/class_info.json', 'w', encoding='utf-8') as f:
            json.dump(class_info, f, ensure_ascii=False, indent=2)
        
        print("✅ Çalışan model kaydedildi!")

def main():
    """Ana fonksiyon."""
    print("🎯 SkinAI - Basit Enhanced Model Training")
    print("=" * 50)
    
    # Veri kontrolü
    data_path = Path("data/processed/train")
    if not data_path.exists():
        print("❌ Veri bulunamadı!")
        return
    
    # Basit enhanced model
    simple_model = SimpleEnhancedModel()
    
    # Eğitim (kısa ve stabil)
    history = simple_model.train_simple(epochs=20)
    
    # Kaydet
    simple_model.save_working_model()
    
    print("\n🎉 Basit enhanced model hazır!")
    print("👉 Web uygulamasını test edin: python app.py")

if __name__ == "__main__":
    main() 