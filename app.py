import streamlit as st
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

st.set_page_config(page_title="Doculy Verifier", page_icon="🛡️")

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return processor, model

st.title("🛡️ Doculy Verifier (Open Source)")

st.write(
    "Upload a certificate image to get a basic forgery-risk signal. "
    "This is an open-source prototype, **not production-grade**."
)

uploaded = st.file_uploader(
    "Upload certificate image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")

    processor, model = load_model()
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    risk = probs[0, 1:].max().item() * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Certificate", width=350)

    with col2:
        st.subheader("Verification Result")
        st.metric("Forgery Risk", f"{risk:.1f}%")
        st.progress(int(risk))

        if risk > 35:
            st.error("High risk – treat as possibly forged. Manual review recommended.")
        else:
            st.success("Low risk – likely genuine, but not guaranteed.")

    st.warning(
        "Note: This prototype uses a general image model and is not trained specifically "
        "on forged certificates. Use this only as a demo signal."
    )
else:
    st.info("Please upload a certificate image to begin verification.")
