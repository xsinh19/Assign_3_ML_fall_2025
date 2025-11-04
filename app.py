import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import re
import os

#
# --- 1. Model Definition (MUST match the training scripts) ---
#
# We must define the class here so PyTorch can load the
# saved state_dict into this structure.
#
class MLPTextGenerator(nn.Module):
    """
    MLP-based N-gram model, matching the one used in the Colab notebooks.
    """
    def __init__(self, vocab_size, embed_dim, context_length, hidden_layers, activation):
        super(MLPTextGenerator, self).__init__()
        
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_hidden = hidden_layers
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Flatten Layer
        self.flatten = nn.Flatten()
        
        # 3. Activation Function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        
        # 4. Hidden Layers
        input_size = context_length * embed_dim
        self.hidden1 = nn.Linear(input_size, 1024)
        
        if self.num_hidden == 2:
            self.hidden2 = nn.Linear(1024, 1024)
            
        # 5. Output Layer
        self.output = nn.Linear(1024, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, context_length)
        x = self.embedding(x)
        # x shape: (batch_size, context_length, embed_dim)
        
        x = self.flatten(x)
        # x shape: (batch_size, context_length * embed_dim)
        
        x = self.activation(self.hidden1(x))
        # x shape: (batch_size, 1024)
        
        if self.num_hidden == 2:
            x = self.activation(self.hidden2(x))
        
        logits = self.output(x)
        # logits shape: (batch_size, vocab_size)
        return logits

#
# --- 2. Helper Functions (Loading & Generation) ---
#

# Use Streamlit's caching to load artifacts only once [10, 16]
@st.cache_resource
def load_vocab_artifacts(dataset_folder):
    """Loads the saved vocabulary artifacts for the chosen dataset."""
    vocab_path = os.path.join(dataset_folder, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        st.error(f"Missing 'vocab.pkl' file in '{dataset_folder}'. Please check your folder structure.")
        return None
    try:
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading 'vocab.pkl' from '{dataset_folder}': {e}")
        return None

@st.cache_resource
def load_dynamic_model(dataset_folder, vocab_size, context, embed, hidden, act):
    """
    Dynamically loads the correct.pth file based on user selection.
   
    """
    # 1. Construct the model filename from parameters
    model_name = f"model_ctx{context}_embed{embed}_hidden{hidden}_act_{act}.pth"
    model_path = os.path.join(dataset_folder, "trained_models", model_name)
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Did you train this combination and place it in the correct folder?")
        return None

    # 2. Instantiate the model structure
    model = MLPTextGenerator(vocab_size, embed, context, hidden, act)
    
    # 3. Load the saved weights (state_dict)
    try:
        # Load weights onto CPU for inference
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def stable_softmax(logits, temperature):
    """
    Numerically stable softmax with temperature.
    [13, 14, 17, 15]
    """
    logits = logits / temperature
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def preprocess_seed_text(seed_text, dataset_choice):
    """
    Applies the correct, dataset-specific preprocessing to the user's input.
    """
    text = seed_text.lower()
    if dataset_choice == "Sherlock Holmes":
        # Matches the Sherlock training script
        text = text.replace('.', '. ')
        text = re.sub(r'[^a-z0-9 \.]', '', text)
    
    elif dataset_choice == "scikit-learn":
        # Matches the sklearn training script
        # [18, 19, 20, 21, 22, 23, 24, 25, 11, 26]
        for char in ['.', '_', '(', ')', '[', ']', '=', ':', ',', '*', '-']:
            text = text.replace(char, f' {char} ')
        text = re.sub(r'[^a-z0-9 \._\(\)\[\]=:,*-]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def generate_text_from_model(model, artifacts, seed_text, num_to_gen, 
                             context_length, temperature, random_seed, dataset_choice):
    """
    Main text generation function with temperature sampling.
    """
    # Extract artifacts
    word_to_int = artifacts['word_to_int']
    int_to_word = artifacts['int_to_word']
    
    # Preprocess the seed text using the correct function
    seed_tokens = preprocess_seed_text(seed_text, dataset_choice)
    generated_text_tokens = list(seed_tokens) # Start with the seed
    
    # Set seed for reproducible generation
    np.random.seed(random_seed)
    
    model.eval()
    with torch.no_grad():
        for i in range(num_to_gen):
            # 1. Prepare input context
            context_tokens = generated_text_tokens[-context_length:]
            
            # 2. Pad if context is too short
            if len(context_tokens) < context_length:
                # Use '.' as the padding token, as defined in training
                pad_list = ['.'] * (context_length - len(context_tokens))
                context_tokens = pad_list + context_tokens
                
            # 3. Convert to integers, handling OOV words
            # This is the OOV handling strategy: map unknown words to <UNK>
            #
            context_ints = []
            for word in context_tokens:
                context_ints.append(word_to_int.get(word, word_to_int['<UNK>']))
            
            # 4. Predict (get logits)
            X_input = torch.tensor([context_ints], dtype=torch.long)
            logits = model(X_input)
            
            # 5. Sample from logits using temperature
            # [13, 14, 17, 27, 15, 28, 29, 30]
            logits_numpy = logits.squeeze().cpu().numpy()
            probabilities = stable_softmax(logits_numpy, temperature)
            next_word_int = np.random.choice(len(probabilities), p=probabilities)
            
            # 6. Convert back to word
            next_word = int_to_word[next_word_int]
            
            # 7. Append to generated text
            generated_text_tokens.append(next_word)
            
    # Return the full string (original seed + generated)
    return " ".join(generated_text_tokens)

#
# --- 3. Streamlit UI ---
#

st.set_page_config(layout="wide")
st.title("MLP Next-Word Predictor")
st.info("This app runs two different sets of models trained on two different datasets. "
        "Select a dataset, then choose the hyperparameters for the specific model you want to test.")

# --- Sidebar (Controls) ---
st.sidebar.header("1. Model & Dataset")

# Master control: Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset Category:",
    ("Sherlock Holmes (Natural Language)", "scikit-learn (Structured Text)")
)

# --- Dynamic Controls based on Dataset Choice ---
if "Sherlock" in dataset_choice:
    dataset_folder = "sherlock_models_and_vocab"
    # Controls based on the Sherlock training script
    context_length = st.sidebar.select_slider(
        "Context Length (N-grams):",
        options=[6, 7, 8, 9],
        help="Select the context window size (6-9) for the Sherlock model."
    )
    # Both scripts used only 64, so this is fixed.
    embed_dim = 64
else:
    dataset_folder = "sklearn_models_and_vocab"
    # Controls based on the sklearn training script
    context_length = st.sidebar.select_slider(
        "Context Length (N-grams):",
        options=[6, 7, 8],
        help="Select the context window size (6-8) for the scikit-learn model."
    )
    # Both scripts used only 64, so this is fixed.
    embed_dim = 64


st.sidebar.header("2. Model Architecture")
# Common controls for both datasets
hidden_layers = st.sidebar.selectbox(
    "Hidden Layers:",
    options=[1, 2],
    index=1 # Default to 2 layers
)
activation = st.sidebar.selectbox(
    "Activation Function:",
    options=["relu", "tanh"]
)

st.sidebar.header("3. Generation Controls")
temperature = st.sidebar.slider(
    "Temperature (Randomness):",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Controls randomness. Lower (e.g., 0.2) = more predictable and repetitive. Higher (e.g., 1.5) = more random and creative."
)
random_seed = st.sidebar.number_input(
    "Random Seed:",
    value=42,
    help="An integer to seed the random sampler for reproducible results."
)
num_to_gen = st.sidebar.number_input(
    "Number of words to generate (k):",
    min_value=10,
    max_value=200,
    value=50
)

# --- Main Page (Input/Output) ---
st.header("Text Generation Input")

# Set a dynamic default seed text
default_seed = "holmes looked at the" if "Sherlock" in dataset_choice else "from sklearn.cluster import"
seed_text = st.text_area(
    "Enter your seed text:",
    default_seed,
    height=100
)

if st.button("Generate Text"):
    # 1. Load vocab
    with st.spinner(f"Loading vocabulary for '{dataset_choice}'..."):
        artifacts = load_vocab_artifacts(dataset_folder)
    
    if artifacts:
        vocab_size = artifacts['vocab_size']
        
        # 2. Dynamically load the selected model
        model_name_str = f"ctx={context_length}, embed={embed_dim}, hidden={hidden_layers}, act={activation}"
        with st.spinner(f"Loading model: {model_name_str}..."):
            model = load_dynamic_model(
                dataset_folder,
                vocab_size, 
                context_length, 
                embed_dim, 
                hidden_layers, 
                activation
            )
        
        if model:
            st.success(f"Model loaded: {model_name_str}")
            
            # 3. Generate text
            with st.spinner("Generating text..."):
                generated_output = generate_text_from_model(
                    model,
                    artifacts,
                    seed_text,
                    num_to_gen,
                    context_length,
                    temperature,
                    random_seed,
                    "Sherlock Holmes" if "Sherlock" in dataset_choice else "scikit-learn"
                )
            
            # 4. Display output
            st.subheader("Generated Text")
            
            # Find length of processed seed to highlight only generated part
            processed_seed_len = len(preprocess_seed_text(seed_text, "Sherlock Holmes" if "Sherlock" in dataset_choice else "scikit-learn"))
            generated_part = " ".join(generated_output.split(" ")[processed_seed_len:])
            
            # Display with original, unprocessed seed text bolded
            st.markdown(f"**{seed_text}** {generated_part}")

st.divider()
st.subheader("How This Works")
st.markdown(
    """
    1.  **Dataset & Model Selection:** You first choose a dataset (Sherlock or scikit-learn). The app then loads the corresponding vocabulary (`vocab.pkl`) and presents the available model hyperparameters (context length, etc.) that were trained.
    2.  **Dynamic Model Loading:** Based on *all* your sidebar selections, the app constructs the exact model filename (e.g., `model_ctx8_embed64_hidden2_act_relu.pth`) and loads it from the correct folder. This is all cached, so it only loads once per configuration.[10]
    3.  **Preprocessing:** Your seed text is cleaned using a *dataset-specific* function. For 'Sherlock', it removes most punctuation. For 'scikit-learn', it *keeps* and *separates* syntactic characters like `.` `_` and `(` to treat them as tokens.
    4.  **OOV Handling:** Any word in your seed text that the model hasn't seen (Out-of-Vocabulary) is automatically mapped to a special `<UNK>` (unknown) token.
    5.  **Temperature Sampling:** The model predicts probabilities for the next word. Instead of just picking the *most likely* word (which is boring), **Temperature** is used to control the randomness of the selection, allowing for more creative or more conservative outputs.[14, 15]
    """
)