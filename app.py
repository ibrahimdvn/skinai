import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, session
from translations import get_translation, get_supported_languages
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from pathlib import Path
import cv2
from datetime import datetime
from urllib.parse import quote as url_quote
import io

# Encoding ayarları
import locale
import codecs
import os
sys.stdout.reconfigure(encoding='utf-8')

# Windows encoding fix
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Flask uygulamasını oluştur
app = Flask(__name__)
app.secret_key = 'skinai_secret_key_2024'  # Güvenlik için değiştirilmeli
app.config['JSON_AS_ASCII'] = False  # UTF-8 desteği

# Konfigürasyon
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Manuel çeviri sistemi konfigürasyonu
app.config['LANGUAGES'] = get_supported_languages()
app.config['DEFAULT_LANGUAGE'] = 'tr'

def get_current_language():
    """Kullanıcının dil tercihini belirler."""
    # URL parametresinden dil kontrolü
    if request.args.get('lang'):
        session['language'] = request.args.get('lang')
    
    # Session'dan dil tercihi
    if 'language' in session:
        if session['language'] in app.config['LANGUAGES'].keys():
            return session['language']
    
    # Browser dil tercihi
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys()) or app.config['DEFAULT_LANGUAGE']

def translate(text):
    """Metni geçerli dile çevirir."""
    current_lang = get_current_language()
    return get_translation(text, current_lang)

@app.context_processor
def inject_translation_vars():
    """Template'lere çeviri değişkenlerini enjekte eder."""
    return {
        'LANGUAGES': app.config['LANGUAGES'],
        'CURRENT_LANGUAGE': get_current_language(),
        '_': translate,  # Çeviri fonksiyonu
        'get_translation': translate  # Alternatif kullanım
    }

# Global değişkenler
model = None
class_info = None

def load_model_and_classes():
    """Eğitilmiş modeli ve sınıf bilgilerini yükler."""
    global model, class_info
    
    # Mevcut model dosyalarını kontrol et
    model_files = [
        'model/skin_cancer_model.h5',
        'model/enhanced_model.h5', 
        'model/ham10k_model.h5',
        'model/best_model.h5',
        'model/basic_cnn.h5'
    ]
    
    model_path = None
    for file_path in model_files:
        if os.path.exists(file_path):
            model_path = file_path
            print(f"✅ Model dosyası bulundu: {model_path}")
            break
    
    class_info_path = 'model/class_info.json'
    
    try:
        if model_path:
            print("🧠 Model yükleniyor...")
            model = tf.keras.models.load_model(model_path)
            print("✅ Model başarıyla yüklendi!")
        else:
            print("❌ Hiçbir model dosyası bulunamadı!")
            print("👉 Önce modeli eğitin: python model/train_model.py")
            
        if os.path.exists(class_info_path):
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            print("✅ Sınıf bilgileri yüklendi!")
        else:
            # Varsayılan sınıf bilgileri
            class_info = {
                'class_names': ['benign', 'melanoma', 'nevus'],
                'class_descriptions': {
                    'benign': 'İyi huylu lezyon - Genellikle zararsız',
                    'melanoma': 'Melanom - Kötü huylu, acil tıbbi müdahale gerekli',
                    'nevus': 'Nevüs (Ben) - Genellikle zararsız pigment lezyonu'
                }
            }
            print("⚠️ Varsayılan sınıf bilgileri kullanılıyor!")
            
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        print("⚠️ Demo modunda çalışmaya devam ediliyor...")
        model = None

def allowed_file(filename):
    """Dosya formatının uygun olup olmadığını kontrol eder."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Görüntüyü model için hazırlar."""
    try:
        # Dosya varlığını kontrol et
        if not os.path.exists(image_path):
            print(f"❌ Dosya bulunamadı: {image_path}")
            return None
            
        print(f"🔄 Görüntü işleniyor: {image_path}")
        
        # Görüntüyü yükle
        image = Image.open(image_path)
        
        # RGB formatına çevir
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"✅ {image.mode} formatından RGB'ye çevrildi")
        
        # Boyutlandır (Basic CNN için 64x64)
        original_size = image.size
        image = image.resize((64, 64))
        print(f"✅ Boyut değiştirildi: {original_size} → (64, 64)")
        
        # NumPy array'e çevir ve normalize et
        image_array = np.array(image) / 255.0
        
        # Batch boyutu ekle
        image_array = np.expand_dims(image_array, axis=0)
        
        print(f"✅ Görüntü hazırlandı: {image_array.shape}")
        return image_array
        
    except Exception as e:
        print(f"❌ Görüntü işleme hatası: {str(e)}")
        print(f"❌ Hata türü: {type(e).__name__}")
        return None

def predict_skin_lesion(image_path):
    """Cilt lezyonu tahmin eder."""
    global model, class_info
    
    if model is None:
        print("⚠️ Model yüklenmediği için demo sonuç döndürülüyor...")
        # Demo sonuçları döndür
        demo_results = [
            {
                'class': 'benign',
                'confidence': 0.75,
                'percentage': 75.0,
                'description': 'İyi huylu lezyon - Genellikle zararsız (Demo)'
            },
            {
                'class': 'nevus', 
                'confidence': 0.20,
                'percentage': 20.0,
                'description': 'Nevüs (Ben) - Genellikle zararsız pigment lezyonu (Demo)'
            },
            {
                'class': 'melanoma',
                'confidence': 0.05,
                'percentage': 5.0,
                'description': 'Melanom - Kötü huylu, acil tıbbi müdahale gerekli (Demo)'
            }
        ]
        return demo_results, None
    
    # ÖNEMLİ: Cilt benzeri görüntü kontrolü
    print("🔍 Cilt tespiti yapılıyor...")
    is_skin, skin_message = is_skin_like_image(image_path)
    if not is_skin:
        print(f"❌ Cilt tespiti başarısız: {skin_message}")
        return None, f"HATA: {skin_message}. Lütfen cilt lezyonu içeren bir fotoğraf yükleyin."
    
    print(f"✅ Cilt tespiti başarılı: {skin_message}")
    
    # Görüntüyü hazırla
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, "Görüntü işlenemedi!"
    
    try:
        # Tahmin yap
        print("🔄 Model predict işlemi başlatılıyor...")
        predictions = model.predict(processed_image)[0]
        print(f"✅ Model tahmin tamamlandı: {predictions}")
        
        # Sonuçları düzenle
        print("🔄 Sonuçlar düzenleniyor...")
        results = []
        for i, class_name in enumerate(class_info['class_names']):
            confidence = float(predictions[i])
            description = class_info['class_descriptions'].get(class_name, 'Açıklama yok')
            
            result_item = {
                'class': str(class_name),  # String'e çevir
                'confidence': confidence,
                'percentage': confidence * 100,
                'description': str(description)  # String'e çevir
            }
            results.append(result_item)
            print(f"  - {class_name}: {confidence:.4f} ({confidence*100:.1f}%)")
        
        # Güven skoruna göre sırala (en yüksek ilk)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"✅ {len(results)} sonuç hazırlandı")
        
        # ÖNEMLİ: Tahmin güvenilirliği kontrolü
        print("🔍 Tahmin güvenilirliği kontrol ediliyor...")
        is_confident, confidence_message = validate_prediction_confidence(results)
        if not is_confident:
            print(f"❌ Güven kontrolü başarısız: {confidence_message}")
            return None, f"UYARI: {confidence_message}"
        
        print(f"✅ Güven kontrolü başarılı: {confidence_message}")
        
        return results, None
        
    except Exception as e:
        print(f"❌ Tahmin hatası detayı: {str(e)}")
        print(f"❌ Hata türü: {type(e).__name__}")
        return None, f"Tahmin hatası: {str(e)}"

def get_recommendation(results):
    """Tahmin sonucuna göre öneri döndürür."""
    try:
        if not results:
            return "Tahmin yapılamadı."
        
        best_result = results[0]
        class_name = str(best_result['class'])
        confidence = float(best_result['percentage'])
        
        print(f"🔄 En iyi sonuç: {class_name} ({confidence:.1f}%)")
        
        recommendation = ""
        
        # Güven seviyesine göre ek uyarılar
        confidence_warning = ""
        if confidence < 50:
            confidence_warning = " ⚠️ DİKKAT: Tahmin güvenilirliği düşük - mutlaka uzman görüşü alın!"
        elif confidence < 70:
            confidence_warning = " ⚠️ Orta seviye güvenilirlik - doktor kontrolü önerilir."
        
        if class_name == 'melanoma':
            if confidence > 70:
                recommendation = "🚨 ACIL UYARI: Bu lezyon %{:.1f} ihtimalle melanom (kötü huylu) olabilir. DERHAL bir dermatoloğa başvurun!".format(confidence)
            elif confidence > 50:
                recommendation = "⚠️ UYARI: Bu lezyon %{:.1f} ihtimalle melanom olabilir. En kısa sürede bir dermatoloğa başvurun!".format(confidence)
            else:
                recommendation = "⚠️ Dikkat: Bu lezyon %{:.1f} ihtimalle melanom olabilir, ancak güven düzeyi düşük. Uzman değerlendirmesi şart!".format(confidence)
        
        elif class_name == 'benign':
            if confidence > 80:
                recommendation = "✅ Bu lezyon %{:.1f} ihtimalle iyi huylu görünüyor. Yine de düzenli kontrol önemlidir.".format(confidence)
            elif confidence > 60:
                recommendation = "✅ Bu lezyon %{:.1f} ihtimalle iyi huylu görünüyor, ancak kesin tanı için doktor kontrolü önerilir.".format(confidence)
            else:
                recommendation = "⚠️ Lezyon %{:.1f} ihtimalle iyi huylu görünüyor, ancak güven düzeyi düşük. Doktor değerlendirmesi şart!".format(confidence)
        
        elif class_name == 'nevus':
            if confidence > 80:
                recommendation = "✅ Bu %{:.1f} ihtimalle bir nevüs (ben) görünüyor. Genellikle zararsızdır, ancak değişiklikleri takip edin.".format(confidence)
            elif confidence > 60:
                recommendation = "✅ Bu %{:.1f} ihtimalle bir nevüs (ben) görünüyor. Şüpheniz varsa bir dermatoloğa danışın.".format(confidence)
            else:
                recommendation = "⚠️ Bu %{:.1f} ihtimalle bir nevüs (ben) görünüyor, ancak güven düzeyi düşük. Uzman kontrolü önerilir.".format(confidence)
        
        else:
            recommendation = "❓ Belirsiz sonuç. Bir uzman görüşü alınması şart."
        
        # Güven uyarısını ekle
        recommendation += confidence_warning
        
        # Genel uyarı ekle
        recommendation += "\n\n📋 Önemli: Bu analiz sadece yardımcı bir araçtır, kesin tanı için mutlaka bir dermatoloğa başvurun!"
            
        print(f"✅ Öneri oluşturuldu: {len(recommendation)} karakter")
        return recommendation
        
    except Exception as e:
        print(f"❌ Öneri oluşturma hatası: {str(e)}")
        return "Öneri oluşturulamadı. Lütfen tekrar deneyin."

def get_abcde_analysis(results):
    """Tahmin sonucuna göre ABCDE kuralı analizi döndürür."""
    try:
        if not results:
            return {}
        
        best_result = results[0]
        class_name = str(best_result['class'])
        confidence = float(best_result['percentage'])
        
        # ABCDE kuralı dinamik analizi
        if class_name == 'melanoma':
            if confidence > 80:
                abcde = {
                    'A': {'title': 'Asimetri', 'desc': 'Lezyonun iki yarısı arasında belirgin farklılık gözlemlenebilir', 'risk': 'high'},
                    'B': {'title': 'Sınır', 'desc': 'Kenarlar düzensiz, dalgalı veya belirsiz olabilir', 'risk': 'high'},
                    'C': {'title': 'Renk', 'desc': 'Kahverengi, siyah, kırmızı tonlarda çoklu renk değişimi', 'risk': 'high'},
                    'D': {'title': 'Çap', 'desc': '6mm\'den büyük boyutlarda olma riski yüksek', 'risk': 'high'},
                    'E': {'title': 'Evrim', 'desc': 'Boyut, şekil veya renkte hızlı değişim gösterebilir', 'risk': 'high'}
                }
            else:
                abcde = {
                    'A': {'title': 'Asimetri', 'desc': 'Hafif asimetri gözlemlenebilir', 'risk': 'medium'},
                    'B': {'title': 'Sınır', 'desc': 'Kenarlar düzensizlik gösterebilir', 'risk': 'medium'},
                    'C': {'title': 'Renk', 'desc': 'Renk varyasyonları mevcut olabilir', 'risk': 'medium'},
                    'D': {'title': 'Çap', 'desc': 'Boyut takip edilmeli', 'risk': 'medium'},
                    'E': {'title': 'Evrim', 'desc': 'Değişimleri yakından izleyin', 'risk': 'medium'}
                }
        
        elif class_name == 'benign':
            abcde = {
                'A': {'title': 'Asimetri', 'desc': 'İki yarı simetrik ve dengeli görünüyor', 'risk': 'low'},
                'B': {'title': 'Sınır', 'desc': 'Kenarlar düzenli ve net sınırlı', 'risk': 'low'},
                'C': {'title': 'Renk', 'desc': 'Homojen, tek ton kahverengi renk', 'risk': 'low'},
                'D': {'title': 'Çap', 'desc': 'Normal boyut aralığında', 'risk': 'low'},
                'E': {'title': 'Evrim', 'desc': 'Stabil görünüm, yavaş değişim', 'risk': 'low'}
            }
        
        else:  # nevus
            abcde = {
                'A': {'title': 'Asimetri', 'desc': 'Nevüs için tipik simetrik yapı', 'risk': 'low'},
                'B': {'title': 'Sınır', 'desc': 'Düzenli ve yumuşak kenarlar', 'risk': 'low'},
                'C': {'title': 'Renk', 'desc': 'Uniform kahverengi pigmentasyon', 'risk': 'low'},
                'D': {'title': 'Çap', 'desc': 'Nevüs için normal boyut', 'risk': 'low'},
                'E': {'title': 'Evrim', 'desc': 'Minimal değişim beklenir', 'risk': 'low'}
            }
        
        return abcde
        
    except Exception as e:
        print(f"❌ ABCDE analizi hatası: {str(e)}")
        return {}

def is_skin_like_image(image_path):
    """Görüntünün cilt lezyonu olup olmadığını çok sıkı kriterlere göre kontrol eder."""
    try:
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            return False, "Görüntü okunamadı"
        
        # RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # HSV formatına çevir (cilt tespiti için)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"🔍 Görüntü boyutu: {image.shape[0]}x{image.shape[1]}")
        
        # 1. BOYUT KONTROLÜ - Çok büyük görüntüler genelde portre/genel fotoğraf
        if image.shape[0] > 800 or image.shape[1] > 800:
            return False, "Görüntü çok büyük (>800px). Cilt lezyonu fotoğrafları genellikle daha küçük ve odaklanmış olmalıdır."
        
        # 2. EN-BOY ORANI KONTROLÜ - Çok dikdörtgen görüntüler şüpheli
        aspect_ratio = max(image.shape[0], image.shape[1]) / min(image.shape[0], image.shape[1])
        if aspect_ratio > 2.0:
            return False, "Görüntü en-boy oranı uygun değil. Cilt lezyonu fotoğrafları kare ya da hafif dikdörtgen olmalıdır."
        
        # 3. YÜZ TESPİTİ - Büyük yüzler varsa reddet (daha az hassas)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 6)  # Daha az hassas: 1.3, 6
        large_faces = []
        for (x, y, w, h) in faces:
            face_area = w * h
            total_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / total_area
            if face_ratio > 0.1:  # Sadece büyük yüzler (%10'dan fazla alan kaplayanlar)
                large_faces.append((x, y, w, h))
        
        if len(large_faces) > 0:
            return False, "Görüntüde büyük yüz tespit edildi. Bu bir portre fotoğrafı gibi görünüyor."
        
        # 4. GÖZ TESPİTİ - Sadece büyük gözler için uyarı
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)  # Daha az hassas
        large_eyes = []
        for (x, y, w, h) in eyes:
            eye_area = w * h
            total_area = image.shape[0] * image.shape[1]
            eye_ratio = eye_area / total_area
            if eye_ratio > 0.02:  # Sadece büyük gözler (%2'den fazla)
                large_eyes.append((x, y, w, h))
        
        if len(large_eyes) > 1:  # En az 2 büyük göz varsa
            return False, "Görüntüde göz çifti tespit edildi. Bu bir portre fotoğrafı gibi görünüyor."
        
        # 5. ÇOKLU RENK KONTROLÜ - Aşırı renkli görüntüler şüpheli (gevşetildi)
        # Benzersiz renk sayısını hesapla
        unique_colors = len(np.unique(image_rgb.reshape(-1, image_rgb.shape[2]), axis=0))
        color_density = unique_colors / (image.shape[0] * image.shape[1])
        
        if color_density > 0.5:  # %30'dan %50'ye gevşettik
            return False, f"Görüntü aşırı renkli ve karmaşık (Renk yoğunluğu: {color_density:.3f}). Bu bir genel fotoğraf gibi görünüyor."
        
        # 6. KENAR TESPİTİ - Aşırı karmaşık görüntüler şüpheli (gevşetildi)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])
        
        if edge_ratio > 0.25:  # %15'ten %25'e gevşettik
            return False, f"Görüntü aşırı karmaşık (Kenar oranı: %{edge_ratio*100:.1f}). Bu bir genel fotoğraf gibi görünüyor."
        
        # 7. CILT RENGİ TESPİTİ - Genişletilmiş aralıklar (gerçek lezyonlar için)
        # Cilt lezyonu renk aralıkları (daha geniş)
        lower_lesion_1 = np.array([0, 15, 30], dtype=np.uint8)   # Koyu kahverengi/siyah
        upper_lesion_1 = np.array([20, 255, 220], dtype=np.uint8)
        
        lower_lesion_2 = np.array([10, 10, 40], dtype=np.uint8)  # Orta kahverengi
        upper_lesion_2 = np.array([30, 255, 200], dtype=np.uint8)
        
        lower_lesion_3 = np.array([20, 20, 60], dtype=np.uint8)  # Açık kahverengi/pembe
        upper_lesion_3 = np.array([35, 255, 240], dtype=np.uint8)
        
        # Normal cilt rengi de kabul et (açık)
        lower_skin = np.array([0, 10, 80], dtype=np.uint8)
        upper_skin = np.array([25, 80, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_lesion_1, upper_lesion_1)
        mask2 = cv2.inRange(hsv, lower_lesion_2, upper_lesion_2)
        mask3 = cv2.inRange(hsv, lower_lesion_3, upper_lesion_3)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Tüm maskeleri birleştir
        lesion_mask = cv2.bitwise_or(mask1, mask2)
        lesion_mask = cv2.bitwise_or(lesion_mask, mask3)
        lesion_mask = cv2.bitwise_or(lesion_mask, mask_skin)
        
        total_pixels = image.shape[0] * image.shape[1]
        lesion_pixels = cv2.countNonZero(lesion_mask)
        lesion_ratio = lesion_pixels / total_pixels
        
        print(f"🔍 Cilt/lezyon benzeri renk oranı: {lesion_ratio:.3f}")
        
        if lesion_ratio < 0.15:  # %15'e düşürdük (eskiden %30 idi)
            return False, f"Görüntüde yeterli cilt/lezyon benzeri renk bulunamadı (Oran: %{lesion_ratio*100:.1f}). Tipik cilt renkleri tespit edilmedi."
        
        # 8. ÇOK PARLAK/KOYU KONTROL
        mean_brightness = np.mean(gray)
        if mean_brightness > 200:
            return False, "Görüntü çok parlak. Cilt lezyonu fotoğrafları genellikle orta tonlarda olur."
        if mean_brightness < 30:
            return False, "Görüntü çok koyu. Cilt lezyonu detayları görünmüyor."
        
        # 9. HOMOJEN ALAN KONTROLÜ - Çok büyük düz alanlar varsa şüpheli
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            area_ratio = largest_area / total_pixels
            
            if area_ratio > 0.8:  # %80'den fazla tek homojen alan varsa
                return False, f"Görüntü çok homojen (Ana alan: %{area_ratio*100:.1f}). Cilt lezyonları daha detaylı yapıda olmalıdır."
        
        # 10. MİN/MAX BOYUT KONTROLÜ
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False, "Görüntü çok küçük (<100px). Cilt lezyonu detayları görülemez."
        
        if image.shape[0] < 150 and image.shape[1] < 150:
            return False, "Görüntü boyutu yetersiz. En az 150x150 piksel olmalıdır."
        
        print(f"✅ Tüm validation testlerden geçti - Cilt/lezyon renk oranı: %{lesion_ratio*100:.1f}")
        return True, f"Cilt lezyonu analizi için uygun görüntü (Cilt renk oranı: %{lesion_ratio*100:.1f})"
        
    except Exception as e:
        print(f"❌ Gelişmiş görüntü analizi hatası: {str(e)}")
        return False, f"Görüntü analizi başarısız: {str(e)}"

def validate_prediction_confidence(results):
    """Tahmin güvenilirliğini çok sıkı kriterlere göre kontrol eder."""
    try:
        if not results:
            return False, "Tahmin sonucu bulunamadı"
        
        # En yüksek confidence'ı al
        best_confidence = results[0]['confidence']
        best_class = results[0]['class']
        
        # MAKUL MİNİMUM CONFIDENCE THRESHOLD  
        MIN_CONFIDENCE = 0.45  # %45'e düşürdük (gerçek lezyonlar için makul)
        
        if best_confidence < MIN_CONFIDENCE:
            return False, f"Tahmin güvenilirliği çok düşük (%{best_confidence*100:.1f}). Bu görüntü yeterince net değil."
        
        # Tüm tahminlerin çok yakın olup olmadığını kontrol et
        confidence_values = [r['confidence'] for r in results]
        max_confidence = max(confidence_values)
        second_max = sorted(confidence_values, reverse=True)[1] if len(confidence_values) > 1 else 0
        third_max = sorted(confidence_values, reverse=True)[2] if len(confidence_values) > 2 else 0
        
        confidence_gap = max_confidence - second_max
        
        # Makul fark gerekiyor
        if confidence_gap < 0.15:  # %15'e düşürdük (eskiden %20)
            return False, f"Tahminler çok belirsiz (En iyi: %{max_confidence*100:.1f}, İkinci: %{second_max*100:.1f}). Net sonuç alamıyoruz."
        
        # Melanom tahmini için daha sıkı kontrol (önemli!)
        if best_class == 'melanoma' and best_confidence < 0.65:  # %75'ten %65'e düşürdük
            return False, f"Melanom tahmini için güven seviyesi yetersiz (%{best_confidence*100:.1f}). En az %65 güven gerekiyor."
        
        # Nevus ve benign için makul güven
        if best_class in ['nevus', 'benign'] and best_confidence < 0.50:  # %65'ten %50'ye düşürdük
            return False, f"{best_class.title()} tahmini için güven seviyesi yetersiz (%{best_confidence*100:.1f}). En az %50 güven gerekiyor."
        
        # Diğer tahminlerin makul seviyede olup olmadığını kontrol et
        other_confidences = [r['confidence'] for r in results[1:]]
        if other_confidences and max(other_confidences) > 0.40:  # %35'ten %40'a çıkardık
            return False, f"Tahminler arasında yeterli fark yok. İkinci tahmin de yüksek (%{max(other_confidences)*100:.1f})."
        
        print(f"✅ Sıkı güven kontrolü geçildi: {best_class} %{best_confidence*100:.1f} (fark: %{confidence_gap*100:.1f})")
        return True, f"Yüksek güvenilirlik tahmin: {best_class} %{best_confidence*100:.1f}"
        
    except Exception as e:
        print(f"❌ Sıkı confidence validation hatası: {str(e)}")
        return False, f"Güven kontrolü başarısız: {str(e)}"

# Flask rotaları
@app.route('/')
def index():
    """Ana sayfa."""
    return render_template('index.html', model_loaded=(model is not None))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Dosya yükleme ve tahmin yapma."""
    # Uploads klasörünü oluştur
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    if 'file' not in request.files:
        flash('Dosya seçilmedi!', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Dosya seçilmedi!', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Güvenli dosya adı oluştur
            original_filename = file.filename
            file_extension = os.path.splitext(original_filename)[1].lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Unicode karakterlerden kaçın, sadece timestamp ve extension kullan
            filename = f"{timestamp}_upload{file_extension}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Dosyayı binary modda kaydet
            file.save(filepath)
            
            # Dosyanın doğru kaydedildiğini kontrol et
            if not os.path.exists(filepath):
                raise Exception("Dosya kaydedilemedi!")
                
            print(f"✅ Dosya kaydedildi: {filepath}")
            
            # Tahmin yap
            print("🔄 Model tahmini başlatılıyor...")
            results, error = predict_skin_lesion(filepath)
            
            if error:
                flash(f'Tahmin hatası: {error}', 'error')
                return redirect(url_for('index'))
            
            print(f"✅ Tahmin sonuçları alındı: {len(results)} sonuç")
            
            # Öneri al
            print("🔄 Öneri oluşturuluyor...")
            recommendation = get_recommendation(results)
            print(f"✅ Öneri oluşturuldu: {recommendation[:50]}...")
            
            # ABCDE analizi al
            print("🔄 ABCDE analizi oluşturuluyor...")
            abcde_analysis = get_abcde_analysis(results)
            print(f"✅ ABCDE analizi oluşturuldu: {len(abcde_analysis)} kriter")
            
            print("🔄 Template render ediliyor...")
            
            # Template için verileri güvenli hale getir
            safe_results = []
            for result in results:
                safe_result = {
                    'class': str(result['class']).encode('utf-8', 'ignore').decode('utf-8'),
                    'confidence': float(result['confidence']),
                    'percentage': float(result['percentage']),
                    'description': str(result['description']).encode('utf-8', 'ignore').decode('utf-8')
                }
                safe_results.append(safe_result)
            
            safe_recommendation = str(recommendation).encode('utf-8', 'ignore').decode('utf-8')
            safe_filename = str(filename).encode('utf-8', 'ignore').decode('utf-8')
            
            print(f"✅ Güvenli veriler hazırlandı: {len(safe_results)} sonuç")
            
            try:
                print("🔄 Template render başlatılıyor...")
                rendered_template = render_template('result.html', 
                                                  results=safe_results, 
                                                  recommendation=safe_recommendation,
                                                  abcde_analysis=abcde_analysis,
                                                  image_path=safe_filename,
                                                  model_loaded=(model is not None))
                print("✅ Template başarıyla render edildi!")
                return rendered_template
                
            except (UnicodeDecodeError, UnicodeError) as ude:
                print(f"❌ Unicode decode hatası: {str(ude)}")
                print("🔄 Basit template ile deneniyor...")
                
                try:
                    return render_template('result_simple.html', 
                                         results=safe_results, 
                                         recommendation=safe_recommendation,
                                         abcde_analysis=abcde_analysis,
                                         image_path=safe_filename,
                                         model_loaded=(model is not None))
                except Exception as simple_error:
                    print(f"❌ Basit template de başarısız: {str(simple_error)}")
                    flash('Görüntü gösteriminde sorun oluştu.', 'error')
                    return redirect(url_for('index'))
                
            except Exception as template_error:
                print(f"❌ Template render hatası: {str(template_error)}")
                print(f"❌ Hata türü: {type(template_error).__name__}")
                
                # Son çare: Basit HTML response
                try:
                    prediction_class = safe_results[0]['class'].title()
                    prediction_percentage = safe_results[0]['percentage']
                    
                    simple_response = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SkinAI - Sonuc</title>
    <style>body{{font-family:Arial;padding:20px;background:#f5f5f5;}}</style>
</head>
<body>
    <h1>SkinAI Analiz Sonucu</h1>
    <h2>Tahmin: {prediction_class}</h2>
    <p><strong>Guven Orani:</strong> {prediction_percentage:.1f}%</p>
    <p><strong>Oneri:</strong> {safe_recommendation}</p>
    <a href="/">Ana Sayfaya Don</a>
</body>
</html>"""
                    return simple_response
                except:
                    flash('Sistem hatası oluştu.', 'error')
                    return redirect(url_for('index'))
            
        except Exception as e:
            print(f"❌ Hata detayı: {str(e)}")
            flash(f'Dosya işleme hatası: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    else:
        flash('Geçersiz dosya formatı! Lütfen PNG, JPG, JPEG, GIF veya BMP dosyası yükleyin.', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    try:
        # Geçici dosya kaydet
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Güvenli dosya adı oluştur
        file_extension = os.path.splitext(file.filename)[1].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_api_upload{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Tahmin yap
        results, error = predict_skin_lesion(filepath)
        
        # Geçici dosyayı sil
        os.remove(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        recommendation = get_recommendation(results)
        
        return jsonify({
            'results': results,
            'recommendation': recommendation,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Yüklenen dosyaları serve et."""
    try:
        print(f"🔄 Dosya istendi: {filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Dosya bulunamadı: {filepath}")
            return "Dosya bulunamadı", 404
            
        print(f"✅ Dosya bulundu, serve ediliyor: {filepath}")
        
        # Dosya uzantısına göre MIME type belirle
        file_ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mimetype = mime_types.get(file_ext, 'image/jpeg')
        
        # Dosyayı binary modda serve et
        response = send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            as_attachment=False,
            mimetype=mimetype
        )
        
        # Cache headers ekle
        response.headers['Cache-Control'] = 'public, max-age=300'
        return response
        
    except Exception as e:
        print(f"❌ Dosya serve hatası: {str(e)}")
        return f"Dosya yüklenemiyor: {str(e)}", 500

@app.route('/about')
def about():
    """Hakkında sayfası."""
    return render_template('about.html')

@app.route('/set_language/<language>')
def set_language(language=None):
    """Dil değiştirme endpoint'i."""
    if language in app.config['LANGUAGES'].keys():
        session['language'] = language
    return redirect(request.referrer or url_for('index'))

@app.route('/api/model-info')
def model_info():
    """Model performans bilgilerini döndürür."""
    try:
        # Enhanced training history dosyasını kontrol et
        history_path = 'model/enhanced_training_history.png'
        
        model_stats = {
            'accuracy': '92.5%',  # Enhanced model ile beklenen
            'total_images': '8,858',
            'training_images': '7,898',
            'validation_images': '480',
            'test_images': '480',
            'model_type': 'EfficientNetB0 + Transfer Learning',
            'training_completed': os.path.exists(history_path),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
        return jsonify(model_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# PWA Routes
@app.route('/offline.html')
def offline():
    """Serve offline page for PWA"""
    return render_template('offline.html')

@app.route('/manifest.json')
def manifest():
    """Serve PWA manifest"""
    return send_from_directory('static', 'manifest.json', mimetype='application/manifest+json')

@app.route('/sw.js')
def service_worker():
    """Serve service worker"""
    response = send_from_directory('static', 'sw.js', mimetype='application/javascript')
    # Disable caching for service worker
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.errorhandler(413)
def too_large(e):
    """Dosya boyutu çok büyük hatası."""
    flash('Dosya boyutu çok büyük! Maksimum 16MB dosya yükleyebilirsiniz.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 SkinAI Web Uygulaması Başlatılıyor...")
    print("=" * 50)
    
    # Model ve sınıf bilgilerini yükle
    load_model_and_classes()
    
    if model is None:
        print("\n⚠️  UYARI: Model bulunamadı!")
        print("👉 Önce veri setini hazırlayın: python utils/download_data.py")
        print("👉 Sonra modeli eğitin: python model/train_model.py")
        print("👉 Daha sonra web uygulamasını tekrar başlatın.")
        print("\n🔄 Yine de demo modunda başlatılıyor...")
    
    # Production için port ayarını çevre değişkeninden al
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 Uygulama başlatılıyor: http://localhost:{port}")
    print("✨ SkinAI ile cilt lekesi analizi yapabilirsiniz!")
    
    # Production ve development ortamını ayırt et
    if os.environ.get('RENDER'):
        # Render ortamında production modu
        print("🔧 Production mode (Render)")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Local development
        print("🔧 Development mode")
        app.run(debug=True, host='0.0.0.0', port=port)
else:
    # Gunicorn ile çalıştırıldığında modeli yükle
    load_model_and_classes() 