import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from keras.models import load_model
model = load_model('./model.h5')

# Classification report data
data = {
    "classes": ['Pepper__bell___Bacterial_spot',
                    'Pepper__bell___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Tomato_Bacterial_spot',
                    'Tomato_Early_blight',
                    'Tomato_Late_blight',
                    'Tomato_Leaf_Mold',
                    'Tomato_Septoria_leaf_spot',
                    'Tomato_Spider_mites_Two_spotted_spider_mite',
                    'Tomato__Target_Spot',
                    'Tomato__Tomato_YellowLeaf__Curl_Virus',
                    'Tomato__Tomato_mosaic_virus',
                    'Tomato_healthy'],
    "precision": [0.99, 0.99, 0.98, 0.83, 0.41, 0.99, 0.74, 1.00, 1.00, 0.91, 0.95, 0.60, 1.00, 0.87, 0.91],
    "recall": [0.99, 0.86, 1.00, 0.99, 1.00, 0.74, 0.90, 0.83, 0.71, 0.97, 0.64, 0.99, 0.98, 0.98, 1.00],
    "f1-score": [0.99, 0.92, 0.99, 0.90, 0.58, 0.85, 0.82, 0.90, 0.83, 0.94, 0.77, 0.74, 0.99, 0.92, 0.95],
    "support": [199, 296, 213, 205, 28, 399, 206, 379, 187, 359, 326, 290, 629, 81, 331]
}

# Create a DataFrame
df = pd.DataFrame(data)

def get_predictions(immg):
    # from skimage import io
    # from keras.preprocessing import image
    import keras.utils as image
    #path='imbalanced/Scratch/Scratch_400.jpg'
    # "C:\Users\prati\Desktop\AML 3104 final project\PlantVillage\Pepper__bell___Bacterial_spot\00f2e69a-1e56-412d-8a79-fdce794a17e4___JR_B.Spot 3132.JPG"
    # pth = immg
    # show_img=image.load_img(pth, grayscale=False, target_size=(200, 200))
    disease_class = data["classes"]

    x = image.img_to_array(immg)
    x = np.expand_dims(x, axis = 0)
    x = np.array(x, 'float32')
    x /= 255
    
    custom = model.predict(x)
    print(custom[0])

    # x = x.reshape([64, 64])

    #plt.gray()
    # plt.imshow(show_img)
    # plt.show()

    a=custom[0]
    ind=np.argmax(a)
            
    print('Prediction:',disease_class[ind])
    return(disease_class[ind])


#User Interface---------------------------------------------------------


pred_flag = False
def main():
    st.label_visibility='collapse'
    st.title('Plant desise identification')
    st.write('# Detecting plant desise using deep learnings dense net model')
    # Display the DataFrame as a table in Streamlit
    st.header('Evaluation Report:')
    

    # Create two columns
    col1, col2 = st.columns(2)

    # Add content to the first column
    with col1:
        st.write('Classification Report:')
        st.table(df)

    # Add content to the second column
    with col2:
        st.write('Confussion matrix')
        st.image("./confussion_matrix.png",width=650)

    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<p style="text-align: center;"><label>Image : </label></p>',unsafe_allow_html=True)
        st.image(image,width=500)
        # image = Image.open()
    if st.button("Predict"):
        resized_image = image.resize((64, 64))
        pred_cls = get_predictions(resized_image)
        st.markdown('<p style="text-align: center;"><label>Prediction : </label></p>',unsafe_allow_html=True)
        # st.image(image,width=900)
        st.markdown(pred_cls)
        # result =''
        # st.success('The output is {}'.format(result))
if __name__ == '__main__': #
    main()