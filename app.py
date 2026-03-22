import streamlit as st
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )
    return processor, model

st.set_page_config(page_title="Doculy Verifier", page_icon="🛡️")
st.title("🛡️ Doculy Verifier (Open Source)")
st.write("Upload a certificate image to get a basic forgery-risk signal. "
         "This is an open-source prototype, **not** production-grade.")

uploaded = st.file_uploader("Upload certificate image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded certificate", use_column_width=True)

    col1, col2 = st.columns(2)
with col1:
    st.image(img, caption="Original", width=300)
with col2:
    st.write("**Risk Score:**")
    st.progress(risk / 100)


    processor, model = load_model()
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Dummy heuristic for now: use max non-class-0 probability as "risk"
    risk = probs[0, 1:].max().item() * 100
    st.metric("Forgery risk (heuristic)", f"{risk:.1f}%")

    if risk > 35:
        st.error("High risk – treat as possibly forged. Manual review recommended.")
    else:
        st.success("Low risk – likely genuine, but not guaranteed.")
