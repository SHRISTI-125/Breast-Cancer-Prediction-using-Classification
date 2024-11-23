import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from PIL import Image
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
df = df.dropna(axis=1)

features = ['radius_mean', 'texture_mean', 'area_mean', 'symmetry_worst']
X = df[features]
y = df['diagnosis']

label = LabelEncoder()
y = label.fit_transform(y)  # 0 for Benign, 1 for Malignant

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lor = LogisticRegression()
lor.fit(X_train, y_train)

def predict_tumor(features):
    features = scaler.transform([features]) 
    prediction = lor.predict(features)
    if prediction == 1:
        return "Malignant"
    else:
        return "Benign"
    

# Interface
st.title("Breast Cancer Prediction App")
st.write("Predicting if the tumor is benign or malignant.")

image = cv2.imread('img.png')

height = 420
aspect_ratio = image.shape[1] / image.shape[0]
width = int(height * aspect_ratio)
resized_image = cv2.resize(image, (width, height))
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

st.image(resized_image_rgb, caption="Breast Cancer Awareness")


#st.image("img.png", use_column_width=True, caption="Breast Cancer Awareness", width=240)
# Slider
st.sidebar.title("Enter Inputs for Prediction")
radius_mean = st.sidebar.number_input("Enter radius mean:", min_value=0.0, value=10.0, step=0.1)
texture_mean = st.sidebar.number_input("Enter texture mean:", min_value=0.0, value=20.0, step=0.1)
area_mean = st.sidebar.number_input("Enter area mean:", min_value=0.0, value=700.0, step=1.0)
symmetry_worst = st.sidebar.number_input("Enter symmetry worst:", min_value=0.0, value=0.1, step=0.01)


st.markdown("""
    <style>
    div.stButton > button {
        background-color: black;
        color: whitesmoke;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #FFD1DF;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

if st.sidebar.button("Predict"):
    input_data = [radius_mean, texture_mean, area_mean, symmetry_worst]
    result = predict_tumor(input_data)
    st.write(f"The predicted diagnosis is: **{result}**")
    if result == "Benign":
        st.write("Benign breast conditions, also known as mammary dysplasia, are usually not a cause for concern and don't increase the risk of breast cancer. Better to visit to Doctor.")
    else:
        st.write("Must to see a doctor immediately.")

if st.sidebar.button("Describe Malignant"):
    st.markdown("""
    <div style="text-align: left;">
        <p><b><h2>Malignant</h2></b></p>Malignant breast cancer is a disease that occurs when abnormal cells in the breast grow out of control and form tumors. 
                Breast cancer can be life-threatening if left untreated.
                <br><h4>Symptoms:</h4>
                <ul>
                <li>Lump in the breast</li>
                <li>Pain in the breast</li>
                <li>Bloody discharge from the nipple and changes in the shape or texture of the nipple.</li>
                <li>Breast or skin over the breast</li>
                </ul>
                <h4>Causes:</h4> Radiation Exposure, Hormone Replacement Treatment, Smoking, Alcohol, Genetic Mutations, etc.,<br>
                <h4>Treatment:</h4>
                <ul>
                <li>Surgery</li><li>Chemotherapy</li><li>Immunotherapy</li>
                <li>Hormone therapy</li><li>Radiation therapy</li>
                </ul>
                <h4>Prevention:</h4>
                <ol>
                <li>Eat a healthy diet</li><li>Do Exercise regularly.</li><li>Avoid beverages containing alcohol</li>
                <li>Mammograms is helpful to detect tumors when they're too small to be felt.</li>
                </ol>
                <h4>Quote</h4>
                <p>“It's not the strength of the body that counts, but the strength of the spirit.” —J.R.R. Tolkien.<br>Stay strong!</p>
    </div>
""", unsafe_allow_html=True)
    
if st.sidebar.button("Describe Benign"):
    st.markdown("""
    <div style="text-align: left;">
        <p><b><h2>Bengin</h2></b></p>Most breast lumps are benign, meaning they are not cancerous. 
                But it is adviced to see a doctor.
                <br><h4>Symptoms:</h4>
                <ul>
                <li>A lump, hard knot, or thickening in the breast, chest, or underarm area</li>
                <li>Pain in the breast</li>
                <li>An itchy, scaly sore or rash</li>
                </ul>
                <h4>Causes:</h4> Hormonal changes, Consuming too much caffeine, Infections, etc.,<br>
                <h4>Treatment:</h4>
                <ul>
                <li>MRI</li><li>Excisional biopsy</li><li>Surgery</li>
                </ul>
                <h4>Prevention:</h4>
                <ol>
                <li>Eat a nutritious diet</li><li>Exercise regularly</li><li>Decreasing caffeine intake like tea, coffee, chocolates</li>
                <li>Get regular mammogram screenings.</li>
                </ol>
                <h4>Quote</h4>
                <p>“Tough times don't last, tough people do.” </p>
    </div>
""", unsafe_allow_html=True)
