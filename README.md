# **🌾 Crop and Fertilizer Recommendation System 🌱**

## **📊 Project Overview**

The **Crop and Fertilizer Recommendation System** is an AI-powered tool designed to assist farmers in making data-driven decisions regarding crop selection and fertilizer usage. By leveraging machine learning models, this system provides accurate recommendations based on environmental and soil conditions, ensuring optimal farming practices for higher yields and sustainable agriculture. 🌾🌍

### **🔑 Key Features**
- **🌱 Crop Recommendation**: Suggests the most suitable crop based on soil and environmental factors like nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, and rainfall.
- **💧 Fertilizer Recommendation**: Recommends the ideal fertilizer based on the crop type and soil characteristics.
- **🖥️ Interactive Web Interface**: Built with **Streamlit**, the app provides an easy-to-use interface for farmers to input data and receive real-time predictions.
- **📈 Model Evaluation**: The system's models are evaluated using precision, recall, F1-score, accuracy, and confusion matrices to ensure robustness and accuracy.

---

## **🛠️ Technologies Used**

- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For training machine learning models (Naive Bayes, Random Forest).
- **Pandas** & **NumPy**: For data manipulation and preprocessing.
- **Matplotlib** & **Seaborn**: For data visualization and statistical plotting.
- **Plotly**: For advanced interactive visualizations.
- **Pickle**: For saving and loading trained machine learning models.
- **Jupyter Notebooks**: For model development, evaluation, and experimentation.

---

## **🚀 Getting Started**

To set up and run this project locally, follow these steps:

### **1. Clone the Repository**

```bash
git clone https://github.com/fox1515/Crop_and_Fertilizer_Recommendation_Model.git
cd Crop_and_Fertilizer_Recommendation_Model
```

### **2. Install Dependencies**

We recommend using a virtual environment. To set up your environment, use the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### **3. Run the Application**

To start the Streamlit app, run:

```bash
streamlit run app.py
```

This will launch the web application in your browser at `http://localhost:8501`. 🌐

---

## **📁 Project Structure**

```
/Crop_and_Fertilizer_Recommendation_Model
├── app.py                    # Main Streamlit web application
├── Crop.pkl             # Saved crop recommendation model (Naive Bayes)
├── fertilizer.pkl       # Saved fertilizer recommendation model (Random Forest)
├── crop_recco.iynb                  # Jupyter notebook for crop recommendation model training
├── Fertilizer.ipynb           # Jupyter notebook for fertilizer recommendation model training
└── requirements.txt           # List of required Python dependencies
```

---

## **📊 Data**

The dataset used for training the recommendation models is stored in `Crop_recommendation.csv`. It contains agricultural data, including features such as:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **temperature**: Average temperature in Celsius
- **humidity**: Average relative humidity
- **ph**: pH level of the soil
- **rainfall**: Annual rainfall (mm)
- **label**: Crop type (target variable)

The dataset is preprocessed to handle missing values, normalize/standardize features, and encode categorical variables for machine learning model training.

---

## **⚙️ How It Works**

### **1. Data Preprocessing**
- The raw data is cleaned and preprocessed, including feature scaling using **MinMaxScaler** and **StandardScaler**.
- Categorical features like **soil_type** and **crop_type** are encoded into numerical values for model compatibility.

### **2. Model Training**
- **🌱 Crop Recommendation**: A **Naive Bayes** classifier is trained to predict the most suitable crop based on soil and environmental factors.
- **💧 Fertilizer Recommendation**: A **Random Forest Classifier** is used to recommend the most appropriate fertilizer based on the crop and soil conditions.

### **3. Model Evaluation**
- Models are evaluated using metrics such as **precision**, **recall**, **F1-score**, and **accuracy**. A **confusion matrix** is also used to visualize prediction errors.

### **4. Deployment**
- The trained models are serialized using **Pickle** and integrated into a **Streamlit** application for real-time predictions. Users can enter soil parameters and receive crop or fertilizer recommendations instantly. 🚜

---

## **🤝 Contributing**

We welcome contributions from the community! Whether you have suggestions for improvements, bug fixes, or new features, feel free to:
1. Fork the repository
2. Create a new branch for your feature/bugfix
3. Open a pull request

### **🐞 Bug Reports & Issues**
To report any issues or bugs, please open an issue on the GitHub repository.

---

## **📜 License**

This project is licensed under the **MIT License**. See the LICENSE file for more details.

---

## **🙏 Acknowledgments**

- **Machine Learning Algorithms**: We used **Naive Bayes** and **Random Forest** to train the crop and fertilizer recommendation models.
- **Data Science Libraries**: Special thanks to libraries such as **Scikit-learn**, **Pandas**, and **Seaborn** for providing essential tools for data analysis and machine learning.
- **Streamlit**: For making the deployment of machine learning models quick and easy with an interactive interface.

---
