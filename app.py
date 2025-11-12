import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import joblib

# ---------------- CONFIG ----------------
MODEL_DIR = "model_files"
MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- SET PAGE CONFIG ----------------
st.set_page_config(page_title="💊 Drug Review Analyzer", layout="wide", page_icon="💊")

# ---------------- LOAD COMPONENTS ----------------
@st.cache_resource
def load_all():
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/tokenizer")
    condition_encoder = joblib.load(f"{MODEL_DIR}/condition_encoder.pkl")
    sentiment_encoder = joblib.load(f"{MODEL_DIR}/sentiment_encoder.pkl")

    with open(f"{MODEL_DIR}/config.json", "r") as f:
        config = json.load(f)

    num_conditions = len(condition_encoder.classes_)
    num_sentiments = len(sentiment_encoder.classes_)

    class MultiTaskBERT(nn.Module):
        def __init__(self, model_name, num_conditions, num_sentiments):
            super(MultiTaskBERT, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(0.3)
            self.condition_head = nn.Linear(self.bert.config.hidden_size, num_conditions)
            self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiments)
            self.rating_head = nn.Linear(self.bert.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
            cond_logits = self.condition_head(pooled)
            sent_logits = self.sentiment_head(pooled)
            rating_pred = self.rating_head(pooled).squeeze()
            return cond_logits, sent_logits, rating_pred

    model = MultiTaskBERT(MODEL_NAME, num_conditions, num_sentiments)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/multi_task_bert.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return tokenizer, condition_encoder, sentiment_encoder, model

tokenizer, condition_encoder, sentiment_encoder, model = load_all()

# ---------------- HEADER ----------------
st.title("💊 Drug Review Analyzer")
st.subheader("Predict condition, sentiment, and rating from a drug review using BERT")

# ---------------- INPUT AREA ----------------
col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area(
        "🗒️ Enter a drug review below:",
        placeholder="Example: This medicine helped with my anxiety but caused drowsiness.",
        height=150
    )

with col2:
    st.markdown("### ⚙️ Model Info")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write(f"**Device:** {'GPU ⚡' if torch.cuda.is_available() else 'CPU 🧠'}")

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Review"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        with st.spinner("Analyzing the review..."):
            inputs = tokenizer(
                review_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                cond_logits, sent_logits, rating_pred = model(
                    inputs["input_ids"], inputs["attention_mask"]
                )

            cond_idx = torch.argmax(cond_logits, dim=1).item()
            sent_idx = torch.argmax(sent_logits, dim=1).item()
            rating_val = rating_pred.item()

            predicted_condition = condition_encoder.inverse_transform([cond_idx])[0]
            predicted_sentiment = sentiment_encoder.inverse_transform([sent_idx])[0]

            # Round & clip rating
            predicted_rating = int(round(rating_val))
            predicted_rating = np.clip(predicted_rating, 1, 10)

        # ---------------- DISPLAY RESULTS ----------------
        st.success("✅ Analysis Complete!")
        st.write("### 🧠 Prediction Results")
        st.write(f"**🩺 Predicted Condition:** {predicted_condition}")
        st.write(f"**💬 Sentiment:** {predicted_sentiment}")
        st.write(f"**⭐ Rating Prediction:** {predicted_rating}/10")

        # Friendly feedback based on sentiment category
        if predicted_sentiment == "Very Effective":
            st.success("🌟 The drug seems **very effective** according to this review!")
        elif predicted_sentiment == "Effective":
            st.info("👍 The review indicates the drug works quite well.")
        elif predicted_sentiment == "Moderate":
            st.warning("😐 The user found the drug moderately effective.")
        elif predicted_sentiment == "Ineffective":
            st.error("⚠️ The drug appears to have **limited effectiveness**.")
        elif predicted_sentiment == "Very Ineffective":
            st.error("❌ The review suggests the drug was **not effective** at all.")
        else:
            st.info("🤔 Sentiment could not be clearly determined.")

        st.balloons()

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using BERT and Streamlit")