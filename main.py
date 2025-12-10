import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
import streamlit as st

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

PEFT_DIR = "./finetuned_flan_t5_en_fr"


# 1) Lire la config PEFT (LoRA) depuis le dossier local
peft_config = PeftConfig.from_pretrained(PEFT_DIR, local_files_only=True)

# 2) Charger le modèle de base utilisé pendant le fine-tuning
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    peft_config.base_model_name_or_path
)

# 3) Appliquer les poids LoRA entraînés
model = PeftModel.from_pretrained(
    base_model,
    PEFT_DIR,
    local_files_only=True,
)

model.to(DEVICE)
model.eval()

# 4) Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

INSTRUCTION = "Translate English to French: "

def translate_sentence(sentence, model, tokenizer, max_length=256):
    model.eval()
    inputs = tokenizer(
        INSTRUCTION + sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Titre et présentation du modèle
st.title("Traduction Anglais → Français (LoRA + Flan-T5)")
st.markdown("""
Ce modèle a été **fine-tuné** à partir de **Flan-T5** sur les datasets **OPUS** et **Europarl** pour la traduction anglais-français.
Il utilise une approche **LoRA (Low-Rank Adaptation)** pour une adaptation légère et efficace.
""")

# Interface utilisateur avec deux colonnes
col1, col2 = st.columns(2)

with col1:
    st.header("Anglais")
    english_text = st.text_area("Entrez votre texte en anglais ici :", height=300, key="english_input")

with col2:
    st.header("Français")
    # Affichage du résultat en texte non modifiable
    if english_text:
        french_text = translate_sentence(english_text, model, tokenizer)
        st.text(french_text)  # Texte non modifiable
    else:
        st.text("")  # Zone vide si aucun texte n'est saisi

# Pied de page
st.markdown("---")
st.caption("© 2025 - Modèle de traduction fine-tuné avec LoRA")