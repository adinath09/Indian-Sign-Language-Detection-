

# Indian Sign Language Detection 🤟

A real-time web application to detect and recognize **Indian Sign Language (ISL)** gestures using computer vision and deep learning.

---

## 📌 Features

* ✅ Real-time hand gesture recognition via webcam
* ✅ Trained CNN-based model on ISL alphabet
* ✅ Web interface built using Flask
* ✅ Modular code structure (training, camera, inference)
* ✅ Interactive UI with HTML templates and static styling

---

🖼️ Screenshot


First _module :



![First_Module](https://github.com/user-attachments/assets/62925a7f-2d10-4258-8fec-84df36ddf1c9)




Second Module :



![Second_module](https://github.com/user-attachments/assets/1270cdde-f61c-40a2-b386-47bdc2d4241e)


## 🗂️ Folder Structure

```
Indian-Sign-Language-Detection-/
│
├── data/                    # ISL image dataset
├── models/                  # Trained deep learning models
├── src/                     # Model training and helper scripts
│   └── ModelTraining.ipynb
├── static/                  # CSS, JS, images for Flask frontend
├── templates/               # HTML pages for Flask
├── app.py                   # Main Flask web application
├── camera.py                # Webcam handling using OpenCV
└── requirements.txt         # Required packages
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/adinath09/Indian-Sign-Language-Detection-.git
cd Indian-Sign-Language-Detection-
```

### 2. Create & Activate a Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Then open your browser and go to: `http://127.0.0.1:5000`

---

## ⚙️ How It Works

1. **User opens the web app.**
2. **Webcam starts** using `camera.py`.
3. **Captured frame** is passed to the trained model.
4. **Predicted sign** is shown on the screen.

---

## 🧠 Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* Flask (for web interface)
* HTML + CSS + JS (in `templates` and `static`)

---

## Dataset
- **ISL Dataset**:
  - 35 Classes: A-Z and 1-9.
  - 1,200 grayscale images per class.
  - Adaptive thresholding applied to enhance feature extraction.

| **Class** | **Examples** | **Processed Images** |
|-----------|--------------|----------------------|
| A-Z       | Letters      | Adaptive Thresholding |
| 1-9       | Numbers      | Adaptive Thresholding |









## 📊 Model Training

* Dataset: Collected ISL signs stored in `data/`
* Model: CNN trained using `src/ModelTraining.ipynb , DataCollection.ipynb`
* Saved model: `models/isl_model.h5`

---

## 🙋‍♂️ Author

**Adinath Nage**
[GitHub](https://github.com/adinath09)
[LinkedIn](https://linkedin.com/in/adinathnage)

---

