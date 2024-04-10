import tensorflow as tf
import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import os
from keras.models import load_model,model_from_json

#model = load_model('class.h5')

with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def save_image(img):
    try:
        complete_path = os.path.join("", "image.jpg")
        image = img.resize((224,224))
        image.save(complete_path)
    except Exception as e:
        st.error(f"An error occurred while saving the image: {e}")

def output(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()

st.markdown("<h1 style='text-align: center;'>GardenGuru</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your Own Web App to Identify Fruits and Vegetables</h3>", unsafe_allow_html=True)

uploaded_file=st.file_uploader("Please Enter Your Image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    save_image(image)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        new_img=image.resize((600, 400))
        st.image(new_img)
    with col3:
        st.write(' ')
    centered_style = """
        <style>
        .centered {
            text-align: center;
        }
        </style>
    """
    img = 'image.jpg'
    ans=output(img)
    print(ans)
    st.markdown(f"<h3 style='text-align: center;color: #99d6ff;'>This is a image of : {ans}</h3>", unsafe_allow_html=True)
    if ans in vegetables:
        st.markdown("<h2 style='text-align: center;color: #00ff00'>It's a Vegetable</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center;color: #ff9933'>It's a Fruit</h2>", unsafe_allow_html=True)
