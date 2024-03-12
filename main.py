import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def predict_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score
def main():
    st.title("Brain Tumor Detection with GoogLeNet")

    st.header("About")
    st.write("""
    Inception V3, also known as GoogLeNet, stands as a formidable architecture in the realm of convolutional neural networks (CNNs), revered for its intricate design and remarkable performance in various image classification tasks. Leveraging its deep layers and sophisticated inception modules, Inception V3 has found particular utility in medical imaging, notably for the critical task of brain tumor classification. In the medical domain, precise and rapid diagnosis is paramount, and Inception V3 rises to the occasion with its ability to discern between different types of brain tumors with a high degree of accuracy.
    
    In the context of brain tumor classification, Inception V3 is trained on a dataset comprising images of glioma, meningioma, pituitary tumors, as well as images depicting the absence of tumors, referred to as 'notumor'. This diverse dataset is instrumental in enabling the model to learn the intricate features and patterns characteristic of each tumor type, thereby facilitating its ability to make informed classifications.
    
    The training process for Inception V3 involves the utilization of sophisticated data augmentation techniques, a critical component in mitigating overfitting and enhancing the model's generalization capabilities. The ImageDataGenerator class, a staple tool in the arsenal of deep learning practitioners, is employed to preprocess the training and testing datasets. Through rescaling the pixel values of the images to a range between 0 and 1, normalization is achieved, which aids in stabilizing the training process and expedites convergence.
    
    Furthermore, employing techniques such as random rotation, shifting, and flipping, augmentation injects variability into the training data, thereby enriching the model's exposure to diverse perspectives of the tumor images. This augmentation strategy is pivotal in enhancing the model's robustness and its ability to generalize well to unseen data, a crucial requirement in real-world medical applications.
    
    The training and validation datasets are organized into directories, facilitating seamless integration with the flow_from_directory method. This method, an integral component of the Keras deep learning framework, streamlines the process of feeding data into the model during training and validation. By specifying parameters such as target size, batch size, and class mode, the datasets are efficiently prepared for consumption by Inception V3, ensuring optimal training and evaluation performance.
    """)

    st.subheader("Model")
    st.image("Model.png", use_column_width=True)

    st.subheader("ROC Curve")
    st.image("ROC.png", use_column_width=True)
    st.subheader("Accuracy: 99.58 %")

    open_camera_checkbox = st.checkbox("Open Camera")
    if open_camera_checkbox:
        st.sidebar.write("Opening Camera...")
        camera = cv2.VideoCapture(0)
        while open_camera_checkbox:
            ret, image = camera.read()
            class_name, confidence_score = predict_image(image)
            st.image(image, caption=f"Class: {class_name[2:]}, Confidence: {np.round(confidence_score * 100)}%",
                     use_column_width=True)
            open_camera_checkbox = st.checkbox("Stop Camera")
            st.markdown(f"<p style='font-size:30px; color:#000000; font-weight:bold;'>Class: {class_name[2:]}</p>",
                        unsafe_allow_html=True)

    select_image_checkbox = st.checkbox("Select Image")
    if select_image_checkbox:
        st.sidebar.write("Choose a file")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.sidebar.write("File Selected!")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            class_name, confidence_score = predict_image(image)
            st.image(image, caption=f"Class: {class_name[2:]}, Confidence: {np.round(confidence_score * 100)}%",
                     use_column_width=True)
            st.markdown(f"<p style='font-size:30px; color:#000000; font-weight:bold;'>Class: {class_name[2:]}</p>",
                        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
