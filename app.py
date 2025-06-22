import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid

# ===================================================================
# BAGIAN 1: DEFINISIKAN ULANG ARSITEKTUR MODEL
# Arsitektur ini HARUS SAMA PERSIS dengan yang digunakan saat pelatihan.
# ===================================================================
latent_dim = 100
n_classes = 10
embedding_dim = 10
device = torch.device("cpu") # Jalankan di CPU untuk deployment

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# ===================================================================
# BAGIAN 2: MUAT MODEL DAN BUAT UI STREAMLIT
# ===================================================================
# Fungsi untuk memuat model yang telah dilatih
@st.cache_resource
def load_model():
    model = Generator().to(device)
    model.load_state_dict(torch.load('generator_cgan.pth', map_location=device))
    model.eval() # Setel model ke mode evaluasi
    return model

generator = load_model()

# Konfigurasi UI Streamlit
st.set_page_config(layout="wide")
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# Pilihan digit untuk pengguna
selected_digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button(f"Generate Images of digit {selected_digit}"):
    st.subheader(f"Generated images of digit {selected_digit}")

    # Buat 5 gambar
    num_images = 5
    with torch.no_grad():
        # Buat noise acak dan label untuk digit yang dipilih
        z = torch.randn(num_images, latent_dim).to(device)
        labels = torch.LongTensor([selected_digit] * num_images).to(device)
        
        # Hasilkan gambar
        generated_images = generator(z, labels)
        
        # Proses gambar untuk ditampilkan
        generated_images = generated_images * 0.5 + 0.5 # Denormalize
        
        # Tampilkan 5 gambar dalam 5 kolom
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            img = generated_images[i].squeeze().cpu().numpy()
            col.image(img, use_column_width=True, caption=f"Sample {i+1}")