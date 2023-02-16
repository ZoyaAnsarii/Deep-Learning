import numpy as np
import tensorflow as tf
from keras import models
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd

st.header("AI Application to determine the hand drawn digit")
st.write("Draw the digit below")
model = tf.keras.models.load_model('mnist/static/mnist.h5')

# Define a function to classify the drawn image
@st.cache_resource #(allow_output_mutation=True)
def predict(image):
    img = Image.fromarray(image).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255
    prediction = model.predict(img)
    return np.argmax(prediction)

# Draw the canvas for users to draw a digit
canvas = st_canvas(width=280, height=280, drawing_mode='freedraw', background_color= 'black',stroke_color='white', stroke_width=20 )
 
# Classify the drawn digit when the user clicks the 'Classify' button
if st.button('Classify'):
    image = canvas.image_data
    digit = predict(image)
    st.write('The drawn digit is:', digit)
