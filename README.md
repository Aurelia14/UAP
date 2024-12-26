# Room Type Classification Using Image Data

### Overview
This project aims to develop a system to classify different types of rooms based on visual properties using deep learning models, particularly Convolutional Neural Networks (CNN) and Residual Networks (ResNet). The classification model identifies images of various room types and automates categorization based on the visual features present in the images. Pre-trained models such as ResNet are utilized to enhance the accuracy and performance of the image classification task.

---

### Dataset Overview

The dataset used in this project is the **House Rooms Image Dataset** from Kaggle. It contains images categorized into five room types:

- **Living Room**
- **Kitchen**
- **Dining Room**
- **Bedroom**
- **Bathroom**

Each image is labeled according to the type of room it depicts, making it suitable for training and evaluating the classification model. The dataset is already divided into training and testing sets to facilitate better performance evaluation.

**Dataset Link:** [House Room Image Dataset](https://kaggle.com)

---

### Preprocessing and Modeling

#### Preprocessing
- Resizing images to **224x224** pixels.
- Performing data augmentation with a 1:4 ratio.
- Normalizing image pixel values.
- Splitting the dataset into training and testing sets for each model.

#### Modeling

##### **Convolutional Neural Networks (CNN)**

**Classification Report:**
- Accuracy: **50%**
- Best-performing label: **Bedroom (62% accuracy)**
- Worst-performing label: **Kitchen (41% accuracy)**

**Learning Curve:**
- **Training loss:** Decreased from 2.5 to 0.75.
- **Validation loss:** Fluctuated between 1.25 and 1.50.
- **Accuracy:** Improved from 20% to 50%.

##### **Residual Networks (ResNet)**

**Classification Report:**
- Accuracy: **85%**
- Best-performing label: **Bedroom (94% accuracy)**
- Worst-performing label: **Kitchen (78% accuracy)**

**Learning Curve:**
- **Training loss:** Decreased from 0.9 to 0.1.
- **Validation loss:** Fluctuated between 0.6 and 1.4.
- **Accuracy:** Improved from 60% to 80%.

---

### Local Web Deployment

#### Features:
- **Home Page:**
  Displays the interface for uploading images.

- **Prediction Page:**
  Shows predictions and classification results after uploading an image.

#### Tools Used:
The implementation is done using Keras and TensorFlow, leveraging pre-trained models to enhance performance and reduce training time.

---

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- OpenCV

---

### Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Aurelia14/klasifikasi-image
   ```
2. Navigate to the project directory:
   ```bash
   cd UAP-ML
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

---

### Author
**Adelta Aurelianti**  
Student ID: 202110370311213
