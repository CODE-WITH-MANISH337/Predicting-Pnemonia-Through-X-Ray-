import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pickle
import cv2

model=tf.lite.Interpreter(model_path='model_optimized.tflite')
model.allocate_tensors()


st.title("Prediting Pnemonia",)
st.subheader('Through X-Ray Predicting Pnemonia')
uploaded_file = st.file_uploader(
    "Choose an X-ray image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load as PIL image
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded X-ray", use_container_width=True)

    st.write("Processing image...")

    
    image=pil_image.resize((224,224))
    
    image=np.array(image,dtype=model.get_input_details()[0]['dtype'])
    image=image/255.0
    image=np.expand_dims(image,axis=0)
    
    
    if st.button('Predict'):
            

            model.set_tensor(model.get_input_details()[0]['index'],image)
            model.invoke()
            prediction=model.get_tensor(model.get_output_details()[0]['index'])[0]
            prediction=(prediction>0.5).astype('int32')
            if prediction==0:
                st.success('You Are Normal')
            if prediction==1:
                st.error('Pnemonia Detected Consultant to Doctor')    
            st.write("🔗 **Find me on:**")
            st.link_button("GitHub","https://github.com/CODE-WITH-MANISH337")
            st.link_button("LinkedIn","https://www.linkedin.com/in/manish-jaiswar-13615b377/?skipRedirect=true")
    
               

                 

        
