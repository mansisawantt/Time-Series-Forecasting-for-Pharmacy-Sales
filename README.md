# Medicine Sales Prediction

## 📌 Project Overview
This project aims to forecast medicine sales using **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** models. The models are trained on past sales data and predict future sales trends for different medicine categories.

## 📁 Project Structure
```
Medicine_sales_prediction/
│── data/
│   ├── preprocessed/               # Contains cleaned and preprocessed sales data
│── models/                         # Stores trained SARIMA models
│── src/
    ├── __init__.py
│   ├── model.py                    # Model training and saving script
    ├── preprocess.py 
│── templates/
│   ├── index.html                   # Frontend interface for prediction
│── app.py                           # Flask web application
│── main.py                          # Script to train models
│── requirements.txt                 # Python dependencies
│── README.md                        # Project documentation
```

##  Installation
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo/MedicineSalesPrediction.git
cd MedicineSalesPrediction
```

### 2️⃣ **Create and Activate Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

##  Usage
### **1️⃣ Train the Models**
Run the following command to train SARIMA models for different medicine categories:
```sh
python main.py
```
This will create trained models inside the `models/` directory.

### **2️⃣ Start the Flask Web App**
To launch the web interface for making predictions:
```sh
python app.py
```
The application will be accessible at: **`http://127.0.0.1:5000/`**

### **3️⃣ Make Predictions**
- Open the web app in your browser.
- Select a medicine category.
- Enter the number of forecast periods (months) and click **Predict**.


---
🚀 **Developed by [Manasi Sawant]** | 📧 Contact: mansisawant438@gmail.com

