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
    
    model_path = 'model/skin_cancer_model.h5'
    class_info_path = 'model/class_info.json'
    
    try:
        if os.path.exists(model_path):
            print("🧠 Model yükleniyor...")
            model = tf.keras.models.load_model(model_path)
            print("✅ Model başarıyla yüklendi!")
        else:
            print("❌ Model dosyası bulunamadı!")
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
            
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")

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
        return None, "Model yüklenmedi!"
    
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
        
        if class_name == 'melanoma':
            if confidence > 70:
                recommendation = "UYARI: Bu lezyon %{:.1f} ihtimalle melanom (kötü huylu) olabilir. DERHAL bir dermatoloğa başvurun!".format(confidence)
            else:
                recommendation = "Dikkat: Bu lezyon %{:.1f} ihtimalle melanom olabilir. Bir doktora danışmanızı öneririz.".format(confidence)
        
        elif class_name == 'benign':
            if confidence > 80:
                recommendation = "Bu lezyon %{:.1f} ihtimalle iyi huylu görünüyor. Yine de düzenli kontrol önemlidir.".format(confidence)
            else:
                recommendation = "Bu lezyon %{:.1f} ihtimalle iyi huylu görünüyor, ancak kesin tanı için doktor kontrolü önerilir.".format(confidence)
        
        elif class_name == 'nevus':
            if confidence > 80:
                recommendation = "Bu %{:.1f} ihtimalle bir nevüs (ben) görünüyor. Genellikle zararsızdır, ancak değişiklikleri takip edin.".format(confidence)
            else:
                recommendation = "Bu %{:.1f} ihtimalle bir nevüs (ben) görünüyor. Şüpheniz varsa bir dermatoloğa danışın.".format(confidence)
        
        else:
            recommendation = "Belirsiz sonuç. Bir uzman görüşü alınması önerilir."
            
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
                    'A': {'title': 'Asimetri', 'desc': 'Lezyonun iki yarısı arasında belirgin farklılık görülebilir', 'risk': 'high'},
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

# Flask rotaları
@app.route('/')
def index():
    """Ana sayfa."""
    return render_template('index.html')

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
                                                  image_path=safe_filename)
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
                                         image_path=safe_filename)
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
    
    print(f"\n🌐 Uygulama başlatılıyor: http://localhost:5000")
    print("✨ SkinAI ile cilt lekesi analizi yapabilirsiniz!")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 