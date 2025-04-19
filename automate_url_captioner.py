import os
import glob
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration #Modèles Blip2

# Charger le processeur et le modèle préentraînés
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Spécifiez le répertoire où se trouvent vos images
image_dir = "/path/to/your/images"
image_exts = ["jpg", "jpeg", "png"]  # spécifiez les extensions de fichier d'image à rechercher

# Ouvrir un fichier pour écrire les légendes
with open("captions.txt", "w") as caption_file:
    # Parcourir chaque fichier image dans le répertoire
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Charger votre image
            raw_image = Image.open(img_path).convert('RGB')

            # Vous n'avez pas besoin de question pour la légende d'image
            inputs = processor(raw_image, return_tensors="pt")

            # Générer une légende pour l'image
            out = model.generate(**inputs, max_new_tokens=50)

            # Décoder les tokens générés en texte
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Écrire la légende dans le fichier, précédée par le nom du fichier image
            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")