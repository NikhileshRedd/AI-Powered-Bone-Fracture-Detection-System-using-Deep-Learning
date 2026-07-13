import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("model.keras")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#eef5ff,#d9e9ff,#f4f7ff);
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#0B3C5D;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:35px;
}

.card{

    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.15);

}

.result-good{

    background:#d4edda;
    color:#155724;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    margin-top:20px;

}

.result-bad{

    background:#f8d7da;
    color:#721c24;
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    margin-top:20px;

}

.footer{

    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:15px;

}

.stButton>button{

    width:100%;
    border-radius:10px;
    height:50px;
    background:#0B74DE;
    color:white;
    font-size:18px;
    border:none;

}

.stButton>button:hover{

    background:#095ab3;
    color:white;

}

section[data-testid="stFileUploader"]{

    background:white;
    padding:20px;
    border-radius:15px;
    border:2px dashed #0B74DE;

}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
"""
<div class='title'>
🦴 Bone Fracture Detection System
</div>

<div class='subtitle'>
AI Powered Medical Image Classification using Deep Learning
</div>
""",
unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------

col1, col2 = st.columns([1,1])

with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload X-Ray Image",
        type=["jpg","jpeg","png"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )

# -----------------------------
# PREDICTION
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.resize((64,64))

    img = np.array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing X-Ray..."):

        time.sleep(1)

        prediction = model.predict(img)

    value = prediction[0][0]

    st.markdown("---")

    if value > 0.5:

        st.markdown("""
        <div class='result-good'>
        ✅ Prediction : NOT FRACTURED
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result-bad'>
        ❌ Prediction : FRACTURED
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(value))

    st.metric(
        "Prediction Confidence",
        f"{value*100:.2f}%"
    )

# -----------------------------
# INFORMATION SECTION
# -----------------------------

st.markdown("---")

st.markdown("""
### About this Project

This application uses a Deep Learning model to classify X-ray images into:

- ✅ Not Fractured
- ❌ Fractured

Upload an X-ray image and the model will analyze it automatically.

""")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("""
<div class='footer'>
Developed using ❤️ Streamlit | TensorFlow | Python
</div>
""", unsafe_allow_html=True)