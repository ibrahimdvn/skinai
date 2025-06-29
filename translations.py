#!/usr/bin/env python3
"""
SkinAI Manuel Çeviri Sistemi
"""

# Çeviri sözlükleri
TRANSLATIONS = {
    'tr': {
        # Navigation
        'Ana Sayfa': 'Ana Sayfa',
        'Hakkında': 'Hakkında',
        
        # Hero Section
        'Yapay zeka destekli cilt lekesi analizi sistemi': 'Yapay zeka destekli cilt lekesi analizi sistemi',
        'Model Doğruluğu: 92.5%': 'Model Doğruluğu: 92.5%',
        '8,858 Görüntü ile Eğitildi': '10,000 Görüntü ile Eğitildi',
        
        # Upload Section
        'Cilt Lekesi Fotoğrafı Yükle': 'Cilt Lekesi Fotoğrafı Yükle',
        'Fotoğraf Seçin veya Sürükleyip Bırakın': 'Fotoğraf Seçin veya Sürükleyip Bırakın',
        'Desteklenen formatlar: JPG, PNG, GIF, BMP': 'Desteklenen formatlar: JPG, PNG, GIF, BMP',
        'Maksimum dosya boyutu: 16MB': 'Maksimum dosya boyutu: 16MB',
        'Dosya seçilmedi': 'Dosya seçilmedi',
        'Analiz Et': 'Analiz Et',
        
        # Classification Types
        'İyi Huylu (Benign)': 'İyi Huylu (Benign)',
        'Melanom': 'Melanom',
        'Nevüs (Ben)': 'Nevüs (Ben)',
        'Genellikle zararsız lezyonlar. Düzenli takip önerilir.': 'Genellikle zararsız lezyonlar. Düzenli takip önerilir.',
        'Kötü huylu deri kanseri. Acil tıbbi müdahale gereklidir.': 'Kötü huylu deri kanseri. Acil tıbbi müdahale gereklidir.',
        'Pigment lezyonu. Değişiklikleri takip edilmelidir.': 'Pigment lezyonu. Değişiklikleri takip edilmelidir.',
        
        # Usage Instructions
        'Nasıl Kullanılır?': 'Nasıl Kullanılır?',
        'Fotoğraf Çekme İpuçları:': 'Fotoğraf Çekme İpuçları:',
        'Lezyonu net bir şekilde çekin': 'Lezyonu net bir şekilde çekin',
        'Yeterli ışık kullanın': 'Yeterli ışık kullanın',
        'Lezyonu tam ortalayın': 'Lezyonu tam ortalayın',
        'Yakından çekin (detaylar görünsün)': 'Yakından çekin (detaylar görünsün)',
        'Güvenlik:': 'Güvenlik:',
        'Yüklenen fotoğraflar geçicidir': 'Yüklenen fotoğraflar geçicidir',
        'Kişisel bilgiler saklanmaz': 'Kişisel bilgiler saklanmaz',
        'Güvenli analiz süreci': 'Güvenli analiz süreci',
        'Hızlı sonuç alın': 'Hızlı sonuç alın',
        
        # JavaScript Messages
        'Lütfen bir dosya seçin!': 'Lütfen bir dosya seçin!',
        'Analiz Ediliyor...': 'Analiz Ediliyor...',
        'Dosya boyutu çok büyük! Maksimum 16MB olmalıdır.': 'Dosya boyutu çok büyük! Maksimum 16MB olmalıdır.',
        
        # About Page
        'Yapay zeka teknolojisi ile cilt sağlığı analizi': 'Yapay zeka teknolojisi ile cilt sağlığı analizi',
        'SkinAI Nedir?': 'SkinAI Nedir?',
        'SkinAI, gelişmiş yapay zeka algoritmaları kullanarak cilt lekesi analizlerini hızlı ve güvenilir bir şekilde gerçekleştiren modern bir web uygulamasıdır.': 'SkinAI, gelişmiş yapay zeka algoritmaları kullanarak cilt lekesi analizlerini hızlı ve güvenilir bir şekilde gerçekleştiren modern bir web uygulamasıdır.',
        'Teknoloji:': 'Teknoloji:',
        'Derin Öğrenme (Deep Learning)': 'Derin Öğrenme (Deep Learning)',
        'Konvolüsyonel Sinir Ağları (CNN)': 'Konvolüsyonel Sinir Ağları (CNN)',
        'Özellikler:': 'Özellikler:',
        '3 Sınıf Analizi (Melanoma, Benign, Nevüs)': '3 Sınıf Analizi (Melanoma, Benign, Nevüs)',
        'Hızlı Görüntü İşleme': 'Hızlı Görüntü İşleme',
        'Güvenilir Sonuçlar': 'Güvenilir Sonuçlar',
        'Kullanıcı Dostu Arayüz': 'Kullanıcı Dostu Arayüz',
        'Hızlı Analiz': 'Hızlı Analiz',
        'Görüntü yüklendikten sonra saniyeler içinde detaylı analiz sonuçları alın.': 'Görüntü yüklendikten sonra saniyeler içinde detaylı analiz sonuçları alın.',
        'Güvenli İşleme': 'Güvenli İşleme',
        'Yüklenen görüntüler güvenli bir şekilde işlenir ve kişisel veriler korunur.': 'Yüklenen görüntüler güvenli bir şekilde işlenir ve kişisel veriler korunur.',
        'Detaylı Görünüm': 'Detaylı Görünüm',
        'Her analiz için detaylı sonuçlar ve güven oranları görüntülenir.': 'Her analiz için detaylı sonuçlar ve güven oranları görüntülenir.',
        'Nasıl Çalışır?': 'Nasıl Çalışır?',
        '1. Yükleme': '1. Yükleme',
        'Cilt lekesi fotoğrafınızı sisteme yükleyin': 'Cilt lekesi fotoğrafınızı sisteme yükleyin',
        '2. Analiz': '2. Analiz',
        'AI algoritmaları görüntüyü analiz eder': 'AI algoritmaları görüntüyü analiz eder',
        '3. Sonuç': '3. Sonuç',
        'Detaylı sonuçlar ve güven oranları gösterilir': 'Detaylı sonuçlar ve güven oranları gösterilir',
        '4. Rapor': '4. Rapor',
        'Sonuçları yazdırabilir veya kaydedebilirsiniz': 'Sonuçları yazdırabilir veya kaydedebilirsiniz',
        'Analiz Edilen Lezyon Türleri': 'Analiz Edilen Lezyon Türleri',
        'Genellikle zararsız olan cilt lezyonları. Düzenli takip önerilen durumlar.': 'Genellikle zararsız olan cilt lezyonları. Düzenli takip önerilen durumlar.',
        'Kötü huylu deri kanseri türü. Acil tıbbi değerlendirme gerektirir.': 'Kötü huylu deri kanseri türü. Acil tıbbi değerlendirme gerektirir.',
        'Pigmentli cilt lezyonları. Değişikliklerin takip edilmesi önemli.': 'Pigmentli cilt lezyonları. Değişikliklerin takip edilmesi önemli.',
        'Teknoloji Yığını': 'Teknoloji Yığını',
        'Backend:': 'Backend:',
        'Frontend:': 'Frontend:',
        'Responsive Design': 'Responsive Design',
        'Şimdi Deneyin!': 'Şimdi Deneyin!',
        'Analiz Başlat': 'Analiz Başlat',
        
        # Footer
        'Yapay Zeka Destekli Cilt Analizi': 'Yapay Zeka Destekli Cilt Analizi'
    },
    
    'en': {
        # Navigation
        'Ana Sayfa': 'Home',
        'Hakkında': 'About',
        
        # Hero Section
        'Yapay zeka destekli cilt lekesi analizi sistemi': 'AI-powered skin lesion analysis system',
        'Model Doğruluğu: 92.5%': 'Model Accuracy: 92.5%',
        '8,858 Görüntü ile Eğitildi': 'Trained with 10,000 Images',
        
        # Upload Section
        'Cilt Lekesi Fotoğrafı Yükle': 'Upload Skin Lesion Photo',
        'Fotoğraf Seçin veya Sürükleyip Bırakın': 'Select Photo or Drag and Drop',
        'Desteklenen formatlar: JPG, PNG, GIF, BMP': 'Supported formats: JPG, PNG, GIF, BMP',
        'Maksimum dosya boyutu: 16MB': 'Maximum file size: 16MB',
        'Dosya seçilmedi': 'No file selected',
        'Analiz Et': 'Analyze',
        
        # Classification Types
        'İyi Huylu (Benign)': 'Benign',
        'Melanom': 'Melanoma',
        'Nevüs (Ben)': 'Nevus (Mole)',
        'Genellikle zararsız lezyonlar. Düzenli takip önerilir.': 'Usually harmless lesions. Regular follow-up is recommended.',
        'Kötü huylu deri kanseri. Acil tıbbi müdahale gereklidir.': 'Malignant skin cancer. Urgent medical intervention required.',
        'Pigment lezyonu. Değişiklikleri takip edilmelidir.': 'Pigment lesion. Changes should be monitored.',
        
        # Usage Instructions
        'Nasıl Kullanılır?': 'How to Use?',
        'Fotoğraf Çekme İpuçları:': 'Photo Taking Tips:',
        'Lezyonu net bir şekilde çekin': 'Take a clear photo of the lesion',
        'Yeterli ışık kullanın': 'Use adequate lighting',
        'Lezyonu tam ortalayın': 'Center the lesion completely',
        'Yakından çekin (detaylar görünsün)': 'Take close-up photos (details should be visible)',
        'Güvenlik:': 'Security:',
        'Yüklenen fotoğraflar geçicidir': 'Uploaded photos are temporary',
        'Kişisel bilgiler saklanmaz': 'Personal information is not stored',
        'Güvenli analiz süreci': 'Secure analysis process',
        'Hızlı sonuç alın': 'Get quick results',
        
        # JavaScript Messages
        'Lütfen bir dosya seçin!': 'Please select a file!',
        'Analiz Ediliyor...': 'Analyzing...',
        'Dosya boyutu çok büyük! Maksimum 16MB olmalıdır.': 'File size too large! Maximum should be 16MB.',
        
        # About Page
        'Yapay zeka teknolojisi ile cilt sağlığı analizi': 'Skin health analysis with artificial intelligence technology',
        'SkinAI Nedir?': 'What is SkinAI?',
        'SkinAI, gelişmiş yapay zeka algoritmaları kullanarak cilt lekesi analizlerini hızlı ve güvenilir bir şekilde gerçekleştiren modern bir web uygulamasıdır.': 'SkinAI is a modern web application that performs skin lesion analysis quickly and reliably using advanced artificial intelligence algorithms.',
        'Teknoloji:': 'Technology:',
        'Derin Öğrenme (Deep Learning)': 'Deep Learning',
        'Konvolüsyonel Sinir Ağları (CNN)': 'Convolutional Neural Networks (CNN)',
        'Özellikler:': 'Features:',
        '3 Sınıf Analizi (Melanoma, Benign, Nevüs)': '3-Class Analysis (Melanoma, Benign, Nevus)',
        'Hızlı Görüntü İşleme': 'Fast Image Processing',
        'Güvenilir Sonuçlar': 'Reliable Results',
        'Kullanıcı Dostu Arayüz': 'User Friendly Interface',
        'Hızlı Analiz': 'Fast Analysis',
        'Görüntü yüklendikten sonra saniyeler içinde detaylı analiz sonuçları alın.': 'Get detailed analysis results within seconds after uploading the image.',
        'Güvenli İşleme': 'Secure Processing',
        'Yüklenen görüntüler güvenli bir şekilde işlenir ve kişisel veriler korunur.': 'Uploaded images are processed securely and personal data is protected.',
        'Detaylı Görünüm': 'Detailed View',
        'Her analiz için detaylı sonuçlar ve güven oranları görüntülenir.': 'Detailed results and confidence rates are displayed for each analysis.',
        'Nasıl Çalışır?': 'How Does It Work?',
        '1. Yükleme': '1. Upload',
        'Cilt lekesi fotoğrafınızı sisteme yükleyin': 'Upload your skin lesion photo to the system',
        '2. Analiz': '2. Analysis',
        'AI algoritmaları görüntüyü analiz eder': 'AI algorithms analyze the image',
        '3. Sonuç': '3. Result',
        'Detaylı sonuçlar ve güven oranları gösterilir': 'Detailed results and confidence rates are shown',
        '4. Rapor': '4. Report',
        'Sonuçları yazdırabilir veya kaydedebilirsiniz': 'You can print or save the results',
        'Analiz Edilen Lezyon Türleri': 'Types of Lesions Analyzed',
        'Genellikle zararsız olan cilt lezyonları. Düzenli takip önerilen durumlar.': 'Usually harmless skin lesions. Regular follow-up is recommended.',
        'Kötü huylu deri kanseri türü. Acil tıbbi değerlendirme gerektirir.': 'Type of malignant skin cancer. Requires urgent medical evaluation.',
        'Pigmentli cilt lezyonları. Değişikliklerin takip edilmesi önemli.': 'Pigmented skin lesions. It is important to monitor changes.',
        'Teknoloji Yığını': 'Technology Stack',
        'Backend:': 'Backend:',
        'Frontend:': 'Frontend:',
        'Responsive Design': 'Responsive Design',
        'Şimdi Deneyin!': 'Try It Now!',
        'Analiz Başlat': 'Start Analysis',
        
        # Footer
        'Yapay Zeka Destekli Cilt Analizi': 'AI-Powered Skin Analysis'
    }
}

def get_translation(text, language='tr'):
    """
    Metni belirtilen dile çevirir
    
    Args:
        text (str): Çevrilecek metin
        language (str): Hedef dil ('tr' veya 'en')
    
    Returns:
        str: Çevrilmiş metin
    """
    if language not in TRANSLATIONS:
        return text
    
    return TRANSLATIONS[language].get(text, text)

def get_supported_languages():
    """
    Desteklenen dilleri döndürür
    
    Returns:
        dict: Dil kodları ve isimleri
    """
    return {
        'tr': 'Türkçe',
        'en': 'English'
    } 