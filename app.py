import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('breast_cancer_classifier.h5')

# App title
st.title("🩺 Breast Cancer Prediction App")
st.write("Upload a breast cell image to check if it's malignant or benign.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))  # Adjust to your model's input shape
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    img_array = img_array / 255.0  # Normalize if model was trained that way

    # Predict
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    st.subheader(f"🔍 Prediction: {class_label}")

    # Give suggestions
    if class_label == "Malignant":
        st.error("⚠️ This looks malignant.")
        st.markdown("""
        ### 📝 Suggested Next Steps:
        - Visit an oncologist immediately.
        - Schedule a biopsy for confirmation.
        - Start treatment as early as possible.
        - Maintain a healthy diet and reduce stress.
        - Join a cancer support group if available.
        """)
    else:
        st.success("✅ This looks benign.")
        st.markdown("""
        ### 💡 Wellness Tips:
        - Regular self-exams and annual screenings.
        - Apply a warm compress or gentle cream for any lump.
        - Avoid stress and follow a balanced diet.
        - Do yoga or light exercises for hormone balance.
        """)

    st.info("This is not a medical diagnosis. Please consult a doctor.")
