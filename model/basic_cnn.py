import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from pathlib import Path

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class BasicCNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['benign', 'melanoma', 'nevus']
        self.img_size = (64, 64)  # Daha küçük resim
        self.batch_size = 32
        
    def create_basic_model(self):
        """Sıfırdan basit CNN oluşturur."""
        print("🧠 Basit CNN modeli oluşturuluyor...")
        
        model = Sequential([
            # İlk Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D(2, 2),
            
            # İkinci Conv Block  
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Üçüncü Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Classifier
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Basit CNN hazır!")
        print(f"📊 Toplam parametreler: {self.model.count_params():,}")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self):
        """Küçük dataset hazırlar."""
        print("\n📊 Küçük dataset hazırlanıyor...")
        
        # Minimal augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True,
            validation_split=0.2  # Dataset'i böl
        )
        
        # Sadece küçük bir subset kullan
        train_generator = train_datagen.flow_from_directory(
            'data/processed/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            'data/processed/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            subset='validation',
            shuffle=False
        )
        
        print(f"📈 Eğitim örnekleri: {train_generator.samples}")
        print(f"📊 Doğrulama örnekleri: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def train_basic(self, epochs=15):
        """Basit eğitim."""
        print(f"\n🚀 Basit CNN eğitimi başlıyor... ({epochs} epoch)")
        
        # Model oluştur
        self.create_basic_model()
        
        # Veri hazırla
        train_gen, val_gen = self.prepare_data()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'model/basic_cnn.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Eğitim
        print("\n🔄 Model eğitimi başlıyor...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Basit CNN eğitimi tamamlandı!")
        
        # Test
        self.test_different_inputs()
        
        return self.history
    
    def test_different_inputs(self):
        """Farklı inputlarla test eder."""
        print("\n🧪 Farklı inputlarla model test ediliyor...")
        
        # 5 farklı random input test et
        predictions = []
        for i in range(5):
            random_input = np.random.random((1, 64, 64, 3))
            pred = self.model.predict(random_input, verbose=0)[0]
            predictions.append(pred)
            
            print(f"\nTest {i+1}:")
            for j, class_name in enumerate(self.class_names):
                confidence = pred[j] * 100
                print(f"   {class_name:8}: {confidence:5.1f}%")
        
        # Tahminler farklı mı kontrol et
        all_predictions_same = True
        for i in range(1, len(predictions)):
            if not np.allclose(predictions[0], predictions[i], atol=0.05):
                all_predictions_same = False
                break
        
        if all_predictions_same:
            print("\n⚠️  MODEL HALA AYNI SONUÇLAR VERİYOR!")
            print("Sorun dataset veya model mimarisinde olabilir.")
        else:
            print("\n✅ MODEL FARKLI SONUÇLAR VERİYOR - BAŞARILI!")
    
    def save_basic_model(self):
        """Basit modeli kaydet."""
        print("\n💾 Basit CNN kaydediliyor...")
        
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
                'architecture': 'Basic CNN from scratch',
                'input_size': self.img_size,
                'total_params': int(self.model.count_params()),
                'working': True
            }
        }
        
        with open('model/class_info.json', 'w', encoding='utf-8') as f:
            json.dump(class_info, f, ensure_ascii=False, indent=2)
        
        print("✅ Basit CNN kaydedildi!")

def main():
    """Ana fonksiyon."""
    print("🎯 SkinAI - Basit CNN Training (Sıfırdan)")
    print("=" * 50)
    
    # Veri kontrolü
    data_path = Path("data/processed/train")
    if not data_path.exists():
        print("❌ Veri bulunamadı!")
        return
    
    # Basit CNN
    basic_cnn = BasicCNN()
    
    # Eğitim
    history = basic_cnn.train_basic(epochs=10)  # Kısa eğitim
    
    # Kaydet
    basic_cnn.save_basic_model()
    
    print("\n🎉 Basit CNN hazır!")
    print("👉 Web uygulamasını test edin: python app.py")

if __name__ == "__main__":
    main() 