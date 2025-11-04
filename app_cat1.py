import streamlit as st
import torch
import torch.nn as nn
import os
import pickle
import re

# =============================
# 1️⃣ Model Definition
# =============================
class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, hidden_layers, activation):
        super(MLPTextGenerator, self).__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_hidden = hidden_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

        input_size = context_length * embed_dim
        self.hidden1 = nn.Linear(input_size, 1024)
        if hidden_layers == 2:
            self.hidden2 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.activation(self.hidden1(x))
        if self.num_hidden == 2:
            x = self.activation(self.hidden2(x))
        return self.output(x)


# =============================
# 2️⃣ Utility Functions
# =============================
@st.cache_resource
def load_vocab(vocab_path="vocab.pkl"):
    """Load vocabulary mappings."""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab["word_to_int"], vocab["int_to_word"], vocab["vocab_size"]

@st.cache_resource
def load_models(model_dir="trained_models", vocab_size=None, embed_dim=64, device="cpu"):
    """Load all trained MLP models from directory."""
    models = {}
    if not os.path.exists(model_dir):
        st.error(f"Model directory not found: {model_dir}")
        return models

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not model_files:
        st.error("No .pth models found in the directory.")
        return models

    for file in model_files:
        try:
            parts = file.replace(".pth", "").split("_")
            context = int([p for p in parts if p.startswith("ctx")][0][3:])
            hidden = int([p for p in parts if p.startswith("hidden")][0][6:])
            act = [p for p in parts if p in ["relu", "tanh"]][0]

            model = MLPTextGenerator(vocab_size, embed_dim, context, hidden, act).to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, file), map_location=device))
            model.eval()
            models[file] = model
        except Exception as e:
            st.warning(f"⚠️ Error loading {file}: {e}")
    return models


def generate_text(model, seed_text, num_words, context_length, word_to_int, int_to_word, device="cpu"):
    """Generate text word-by-word using a trained model."""
    text = seed_text.lower()
    text = re.sub(r'[^a-z0-9 \.]', '', text)
    seed_tokens = text.split()
    pad_token = word_to_int.get('.', 0)
    generated = seed_tokens.copy()

    model.eval()
    with torch.no_grad():
        for _ in range(num_words):
            context = generated[-context_length:]
            if len(context) < context_length:
                context = ['.'] * (context_length - len(context)) + context
            context_ids = [word_to_int.get(w, 0) for w in context]
            x = torch.tensor([context_ids], dtype=torch.long).to(device)
            y_pred = model(x)
            next_id = torch.argmax(y_pred, dim=1).item()
            next_word = int_to_word.get(next_id, "")
            generated.append(next_word)
    return " ".join(generated)


# =============================
# 3️⃣ Streamlit UI
# =============================
st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("🔮 Next-Word Prediction (MLPTextGenerator)")
st.markdown("Using Sherlock Holmes models trained on a 10,000-word vocabulary.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Using device: {device}")

# --- Load vocab
word_to_int, int_to_word, vocab_size = load_vocab()
st.sidebar.info(f"✅ Vocabulary loaded ({vocab_size:,} words)")

# --- Load models
models = load_models(vocab_size=vocab_size, device=device)
if not models:
    st.error("❌ No models loaded. Check your 'trained_models' folder.")
else:
    st.sidebar.success(f"✅ {len(models)} models loaded successfully.")

# --- UI inputs
model_name = st.selectbox("Select model:", list(models.keys()))
seed = st.text_input("Enter your seed text:", "holmes was a man of")
num_words = st.slider("Words to generate:", 5, 100, 30)

# --- Generate
if st.button("🔮 Generate Next Words"):
    model = models[model_name]
    context_match = re.search(r"ctx(\d+)", model_name)
    context_length = int(context_match.group(1)) if context_match else 8
    output = generate_text(model, seed, num_words, context_length, word_to_int, int_to_word, device)
    st.markdown("### ✍️ Generated Text:")
    st.write(output)
