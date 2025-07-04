﻿# 🩺 SkinAI - AI-Powered Skin Lesion Analysis System

An intelligent skin lesion analysis and classification system powered by artificial intelligence. Built with Flask web framework and CNN (Convolutional Neural Network) model for automated skin lesion detection and analysis.

## 🌐 Multi-Language Support

SkinAI now supports multiple languages with a seamless translation system:
- **Turkish (TR)** - Default language
- **English (EN)** - Full translation support
- Modern dropdown language selector with smooth transitions
- Real-time language switching without page reload

## 🎯 Key Features

- **Web-Based Interface**: User-friendly Flask web application
- **CNN Model**: Deep learning model trained with TensorFlow/Keras
- **3-Class Classification**: Melanoma, Benign, Nevus (Mole)
- **Visual Upload**: Drag-and-drop file upload functionality
- **Real-Time Analysis**: Instant prediction and recommendation system
- **Responsive Design**: Mobile-friendly modern interface
- **PWA Support**: Progressive Web App capabilities
- **Multi-Language**: Turkish and English support
- **Modern UI**: Glassmorphism design with gradient backgrounds

## 🔬 Classification Categories

| Category | Description | Risk Level |
|----------|-------------|------------|
| **Melanoma** | Malignant skin cancer | 🔴 High Risk |
| **Benign** | Benign lesions | 🟢 Low Risk |
| **Nevus** | Pigmented lesion (mole) | 🟡 Follow-up Required |

## 📁 Project Structure

```
skin-cancer-detection/
├── 📁 data/                    # Data files
│   ├── 📁 raw/                 # Raw data
│   ├── 📁 processed/           # Processed data
│   └── 📁 ham10k_processed/    # HAM10000 processed data
├── 📁 model/                   # Model files
│   ├── 🧠 enhanced_model.py    # Enhanced model architecture
│   ├── 📊 training_history.png # Training charts
│   ├── 📄 class_info.json     # Class information
│   └── 🐍 train_model.py       # Model training
├── 📁 static/                  # Static files
│   ├── 📁 css/                 # CSS stylesheets
│   ├── 📁 js/                  # JavaScript files
│   ├── 📁 icons/               # PWA icons
│   ├── 📄 manifest.json        # PWA manifest
│   └── 📄 sw.js                # Service worker
├── 📁 templates/               # HTML templates
│   ├── 🌐 base.html            # Base template with i18n
│   ├── 🏠 index.html           # Homepage
│   ├── ℹ️ about.html           # About page
│   ├── 📊 result.html          # Results page
│   └── 📱 offline.html         # Offline page
├── 📁 translations/            # Translation files
│   └── 📁 en/                  # English translations
├── 📁 uploads/                 # Uploaded files
├── 📁 utils/                   # Utility tools
├── 🐍 app.py                   # Flask application
├── 🐍 translations.py          # Translation system
├── 📋 requirements.txt         # Dependencies
├── 🚀 Procfile                 # Deployment config
└── 📖 README.md                # This file
```

## 🚀 Installation and Setup

### 1. Requirements

- Python 3.10+
- pip package manager
- 4GB+ RAM recommended

### 2. Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd skin-cancer-detection

# 2. Install required packages
pip install -r requirements.txt

# 3. Prepare dataset (demo dataset)
python utils/download_data.py

# 4. Train the model (optional)
python model/train_model.py

# 5. Run the web application
python app.py
```

🌐 **Application URL**: web-production-17b30.up.railway.app

## 📊 Model Architecture & Performance

### Model Details
- **Training Data**: 10,000 images
- **Model Accuracy**: 92.5%
- **Architecture**: Enhanced CNN with batch normalization
- **Input Size**: 224x224x3 RGB images
- **Output**: 3-class probability distribution

### Enhanced CNN Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes
])
```

## 🎨 User Guide

### 1. Homepage
- 📤 Upload skin lesion photo
- 🖱️ Drag-and-drop support
- 📋 Supported formats: JPG, PNG, GIF, BMP
- 🌐 Language selection (TR/EN)

### 2. Analysis Process
1. Select or drag-drop image
2. Click "Analyze" button
3. Wait for AI analysis
4. Review detailed results

### 3. Results Page
- 📸 Uploaded image display
- 📊 Classification probabilities
- 💡 AI recommendations and explanations
- 🎯 Confidence scores for each category

## 🌐 Translation System

### Manual Translation System
- **Implementation**: Custom translation dictionary
- **Languages**: Turkish (default), English
- **Coverage**: Complete UI translation
- **Features**: 
  - Session-based language persistence
  - Real-time language switching
  - Modern dropdown language selector
  - Responsive design for all devices

### Adding New Languages
```python
# In translations.py
TRANSLATIONS = {
    'tr': { 'key': 'Turkish text' },
    'en': { 'key': 'English text' },
    'new_lang': { 'key': 'New language text' }
}
```

## 🔧 API Usage

```python
import requests

url = "http://localhost:5000/api/predict"
files = {'file': open('lesion.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## 📱 Progressive Web App (PWA)

SkinAI includes PWA capabilities:
- **Offline Support**: Basic functionality when offline
- **Installable**: Add to home screen on mobile devices
- **Fast Loading**: Service worker caching
- **Responsive**: Works on all device sizes
- **Modern Icons**: High-resolution app icons

## ⚠️ Important Disclaimers

### 🩺 Medical Responsibility Disclaimer
- This system is for **educational purposes only**
- Does **not provide medical diagnosis**
- **Consult a dermatologist** for definitive diagnosis
- Results are for **reference purposes only**
- **Not a substitute** for professional medical advice

## 📈 Technical Stack

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Image Processing**: PIL, OpenCV
- **Data Handling**: NumPy, Pandas
- **Translation**: Custom manual system
- **PWA**: Service Worker, Web App Manifest

## 🛠️ Development

### Testing
```bash
# Test data preparation
python utils/download_data.py

# Test model training
python model/train_model.py

# Test web application
python app.py

# Test translations
python -c "from translations import get_translation; print(get_translation('Ana Sayfa', 'en'))"
```

### Adding New Features
1. Update templates with translation keys
2. Add translations to `translations.py`
3. Test in both languages
4. Update documentation

## 📚 Dataset Information

Inspired by the HAM10000 dataset with custom preprocessing:
- **Original Classes**: 7 classes → reduced to 3 classes
- **Image Size**: 224x224 pixels
- **Training Set**: 10,000 images
- **Demo Dataset**: Auto-generated sample images for testing

### Class Mapping
| Original HAM10000 | SkinAI Class | Description |
|------------------|--------------|-------------|
| mel, bcc, akiec | Melanoma | Malignant |
| bkl, df, vasc | Benign | Benign |
| nv | Nevus | Mole |

## 🚀 Deployment

### Heroku Deployment
The project includes Heroku deployment configuration:
- `Procfile` for web process
- `requirements.txt` with all dependencies
- Static files served by Flask
- Environment variables supported

```bash
# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

### Environment Variables
- `FLASK_ENV`: Set to 'production' for deployment
- `PORT`: Port number (automatically set by Heroku)

## 🤝 Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Add translations if needed
5. Test thoroughly
6. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.

## 👥 Team

SkinAI Development Team
- Machine Learning & Web Development
- UI/UX Design & Data Science
- Internationalization & Accessibility

## 🔮 Future Enhancements

- [ ] Additional language support (Spanish, French, German)
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] User accounts and history
- [ ] Integration with medical databases
- [ ] Real-time model updates

---

⚠️ **WARNING**: Educational project only. Consult healthcare professionals for medical concerns.

📅 **Version**: 2.0.0 | **Date**: 2025 | **Languages**: Turkish, English

🌟 **Features**: AI Analysis, Multi-language, PWA, Responsive Design
