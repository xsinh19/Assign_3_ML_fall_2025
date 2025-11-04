import streamlit as st
import torch
import torch.nn as nn
import re, os, pickle

# ======================
#  Model Definition
# ======================

class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, hidden_layers, activation):
        super(MLPTextGenerator, self).__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_hidden = hidden_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten = nn.Flatten()

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        input_size = context_length * embed_dim
        hidden_size = 1024  # same as training script

        self.hidden1 = nn.Linear(input_size, hidden_size)
        if self.num_hidden == 2:
            self.hidden2 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.activation(self.hidden1(x))
        if self.num_hidden == 2:
            x = self.activation(self.hidden2(x))
        logits = self.output(x)
        return logits


# ======================
#  Helper Functions
# ======================

@st.cache_resource
def load_vocab():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab["word_to_int"], vocab["int_to_word"], vocab["vocab_size"]

@st.cache_resource
def load_models():
    models = {}
    model_dir = "trained_models"
    word_to_int, int_to_word, vocab_size = load_vocab()

    for file in os.listdir(model_dir):
        if not file.endswith(".pth"):
            continue
        path = os.path.join(model_dir, file)

        # Extract hyperparameters from filename
        # Format: model_ctx{C}_embed{E}_hidden{H}_act_{A}.pth
        m = re.match(r"model_ctx(\d+)_embed(\d+)_hidden(\d+)_act_(relu|tanh)\.pth", file)
        if not m:
            continue
        context_len, embed_dim, hidden_layers, act = m.groups()
        context_len, embed_dim, hidden_layers = int(context_len), int(embed_dim), int(hidden_layers)

        model = MLPTextGenerator(vocab_size, embed_dim, context_len, hidden_layers, act)
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        models[file] = (model, context_len, act)

    return models


def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'([.,()=_!?:;\[\]*])', r' \1 ', text)
    text = re.sub(r'[^a-z0-9 \._\(\)\[\]=:,*-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_text(model, seed_text, num_words, context_len, word_to_int, int_to_word):
    model.eval()
    seed_text = preprocess_input(seed_text)
    tokens = seed_text.split()
    pad_token = word_to_int['.']

    with torch.no_grad():
        for _ in range(num_words):
            context = tokens[-context_len:]
            if len(context) < context_len:
                context = ['.'] * (context_len - len(context)) + context

            context_ids = [word_to_int.get(w, word_to_int['<UNK>']) for w in context]
            x = torch.tensor([context_ids], dtype=torch.long)
            y = model(x)
            pred_id = torch.argmax(y, dim=1).item()
            pred_word = int_to_word.get(pred_id, '<UNK>')
            tokens.append(pred_word)

    return " ".join(tokens)


# ======================
#  Streamlit UI
# ======================

st.title("🧠 Next Word Prediction — scikit-learn Text Generator (MLP)")
st.write("This app uses trained MLP models on the scikit-learn documentation corpus.")

word_to_int, int_to_word, vocab_size = load_vocab()
models = load_models()

if not models:
    st.error("No models found in the 'trained_models/' directory.")
else:
    model_name = st.selectbox("Select a trained model", list(models.keys()))
    model, context_len, act = models[model_name]

    seed_text = st.text_input("Enter seed text", "from sklearn.linear_model import")
    num_words = st.slider("Number of words to generate", 5, 50, 20)

    if st.button("Generate"):
        output = generate_text(model, seed_text, num_words, context_len, word_to_int, int_to_word)
        st.subheader("Generated Text")
        st.write(output)

    st.markdown(f"**Model info:** Context={context_len}, Activation={act}")

