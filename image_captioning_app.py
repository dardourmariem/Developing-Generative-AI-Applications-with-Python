import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Charger le processeur et le modèle pré-entraînés
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convertir le tableau numpy en image PIL et convertir en RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Traiter l'image
    inputs = processor(raw_image, return_tensors="pt")

    # Générer une légende pour l'image
    out = model.generate(**inputs,max_length=50)

    # Décoder les tokens générés en texte
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Génération de légendes d'images",
    description="Ceci est une application web simple pour générer des légendes pour des images en utilisant un modèle entraîné."
)

iface.launch()