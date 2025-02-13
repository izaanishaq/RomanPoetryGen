# Streamlit App for AI Poetry Generator
import streamlit as st
import torch
from unidecode import unidecode
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

st.set_page_config(
    layout="wide",
)

col1, col2 = st.columns(2)

with col1:
    st.image("logo.png", width=800, channels="RGB", output_format="auto", use_container_width=False)

st.markdown("""
    <style>
        /* Main background colors */
        .stApp {
            background-color: black;
        }
        
        /* All divs and containers */
        div.stButton > button {
            background-color: black;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        div.stTextInput > div > div {
            background-color: rgba(0, 0, 0, 0.44);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stTextArea > div > div {
            background-color: black;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        header[data-testid="stHeader"] {
            background-color: black !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }    

        
    

        
    </style>
""", unsafe_allow_html=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "Roman-Urdu-Poetry.csv"

df = pd.read_csv(data_path)
text = " ".join(df['Poetry'].apply(lambda x: unidecode(x).lower()))

chars = sorted(set(text))
vocab = {char: i + 2 for i, char in enumerate(chars)}  # +2 for <unk> and <pad>
vocab['<unk>'], vocab['<pad>'] = 0, 1
idx_to_char = {i: char for char, i in vocab.items()}


embedding = nn.Embedding(len(vocab), 128, padding_idx=vocab['<pad>']).to(device)
lstm = nn.LSTM(128, 256, 3, batch_first=True, dropout=0.2).to(device)
linear = nn.Linear(256, len(vocab)).to(device)

checkpoint = torch.load('poetGenModel.pth', map_location=device)
embedding.load_state_dict(checkpoint['embedding'])
lstm.load_state_dict(checkpoint['lstm'])
linear.load_state_dict(checkpoint['linear'])

embedding.eval()
lstm.eval()
linear.eval()


def forward_pass(inputs, hidden=None):
    lstm_out, hidden = lstm(embedding(inputs), hidden)
    return linear(lstm_out[:, -1, :]), hidden


def generate_text(seed_text, gen_len=250, temperature=0.85):
    seed_text = unidecode(seed_text).lower()
    indices = [vocab.get(char, vocab['<unk>']) for char in seed_text]
    input_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    hidden = (torch.zeros(3, 1, 256).to(device), torch.zeros(3, 1, 256).to(device))

    generated_indices = indices
    with torch.no_grad():
        for _ in range(gen_len):
            outputs, hidden = forward_pass(input_tensor, hidden)
            output = outputs / temperature
            probs = torch.softmax(output, dim=-1)
            pred_idx = torch.multinomial(probs, num_samples=1).item()
            generated_indices.append(pred_idx)
            input_tensor = torch.tensor([[pred_idx]]).to(device)

    generated_text = ''.join([idx_to_char.get(i, '') for i in generated_indices])
    return generated_text

with col2:
    st.title("Roman Urdu Poetry Generator")
    st.write("Enter a seed text to generate poetry!")

    # User Input
    seed_text = st.text_input("Seed Text:", value="wo jo tum ")
    gen_len = st.slider("Generated Text Length (in characters):", min_value=50, max_value=1000, value=250, step=10)
    temperature = st.slider("Temperature (Creativity):", min_value=0.5, max_value=1.5, value=0.85, step=0.05)

    # Generate Button
    if st.button("Generate Poetry"):
        with st.spinner("Generating poetry..."):
            generated_poetry = generate_text(seed_text, gen_len, temperature)
        st.text_area("Generated Poetry:", value=generated_poetry, height=300)
