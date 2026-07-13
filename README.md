# AI-Powered-Bone-Fracture-Detection-System-using-Deep-Learning
Developed an AI-powered bone fracture detection system using TensorFlow, Keras, and Convolutional Neural Networks (CNNs) to classify X-ray images as Fractured or Not Fractured.


# 🦴 AI-Powered Bone Fracture Detection System

An AI-powered web application that detects bone fractures from X-ray images using a Convolutional Neural Network (CNN). The application is built with **TensorFlow**, **Keras**, and **Streamlit**, allowing users to upload an X-ray image and receive an instant prediction indicating whether the bone is **Fractured** or **Not Fractured**.

---

## 📌 Features

- Upload X-ray images in JPG, JPEG, or PNG format.
- Automatic image preprocessing and normalization.
- Deep Learning-based bone fracture detection using CNN.
- Real-time prediction with confidence score.
- Interactive and user-friendly Streamlit interface.
- Lightweight and easy to deploy.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow (PIL)
- HTML & CSS (for custom UI)

---

## 📂 Project Structure

```
Bone-Fracture-Detection/
│
├── model.py                 # CNN model training script
├── app.py                   # Streamlit web application
├── model.keras              # Trained model
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── train/                   # Training dataset
└── test/                    # Testing dataset
```

---

## 📥 Dataset

This project uses an X-ray bone fracture image dataset.

**Download the dataset from Kaggle before running the project.**

You can search on Kaggle using keywords such as:

- Bone Fracture Detection Dataset
- Bone X-ray Images
- Fracture Detection Dataset

After downloading, organize the dataset as shown below:

```
dataset/
│
├── train/
│   ├── Fractured/
│   └── Not Fractured/
│
└── test/
    ├── Fractured/
    └── Not Fractured/
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Bone-Fracture-Detection.git
```

### 2. Navigate to the project folder

```bash
cd Bone-Fracture-Detection
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

Run the training script:

```bash
python model.py
```

This will train the CNN model and generate:

```
model.keras
```

---

## ▶️ Run the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application in your browser.

---

## 🧠 Model Workflow

1. Upload an X-ray image.
2. Resize the image to 64 × 64 pixels.
3. Normalize pixel values.
4. Pass the image through the trained CNN model.
5. Predict whether the bone is **Fractured** or **Not Fractured**.
6. Display the prediction along with the confidence score.

---

## 📊 Future Improvements

- Implement Transfer Learning (EfficientNet, MobileNetV2, ResNet50)
- Improve model accuracy with Data Augmentation
- Add Grad-CAM visualization for explainable AI
- Support multiple bone types
- Deploy the application on Streamlit Cloud or Azure
- Add PDF report generation for predictions

---

## 📸 Application Preview

Add screenshots of the application here.

Example:

```
images/homepage.png
images/prediction.png
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📄 License

This project is intended for educational and research purposes only. It is **not** a certified medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## 👨‍💻 Author

**Nikhilesh Reddy Gondesi**

If you found this project helpful, consider giving it a ⭐ on GitHub!
