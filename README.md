# Handwritten Digit Recognition 

## 📄 Project Overview
This project is the final task of my **Machine Learning Internship at Codmetric**. The objective was to develop a **Deep Learning** model (Neural Network) capable of identifying handwritten digits from $0$ to $9$ using the world-renowned **MNIST dataset**.

## 📊 Dataset Description
The **MNIST (Modified National Institute of Standards and Technology)** dataset consists of:
- **60,000 Training images** and **10,000 Testing images**.
- Each image is a $28 \times 28$ pixel grayscale representation of a handwritten digit.

## 🛠️ Requirements & Methodology
In compliance with the internship requirements, the following implementation steps were taken:

1. **Data Preprocessing:**
   - Loaded the dataset using the `Keras` library.
   - **Normalization:** Scaled the pixel values from $[0, 255]$ to the range $[0, 1]$ to optimize the training process.

2. **Model Architecture:**
   - Designed a **Sequential Neural Network** using **TensorFlow/Keras**.
   - **Input Layer:** `Flatten` layer to convert 2D images into 1D vectors.
   - **Hidden Layer:** `Dense` layer with $128$ neurons and `ReLU` activation.
   - **Regularization:** `Dropout` layer ($20\%$) to prevent overfitting.
   - **Output Layer:** `Dense` layer with $10$ neurons using `Softmax` activation for multi-class classification.

3. **Performance Evaluation:**
   - Compiled the model using the `Adam` optimizer and `Sparse Categorical Crossentropy` loss function.
   - Monitored progress through **Accuracy** and **Loss** plots over $5$ epochs.

4. **Testing on Sample Images:**
   - Verified the model's predictive power by testing it on unseen sample images from the test set and displaying the predicted vs. actual digits.

## 🚀 Results
- **Test Accuracy:** 97.64%
- **Visualization:** Successfully generated training progress plots and confirmed accurate predictions on multiple sample images.

## 💻 Tech Stack
- **Frameworks:** TensorFlow, Keras
- **Libraries:** NumPy, Matplotlib
- **Development Tool:** VS Code
