import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter
from model import combined_loss, count_loss # Stelle sicher, dass model.py da ist!

# --- KONFIGURATION ---
MODEL_PATH = 'best_model.keras'
INPUT_IMAGE = '1a.JPG'
OUTPUT_IMAGE = 'resultat_zaehlung.jpg'
TARGET_DIM = 2000 
TILE_SIZE = 256

def find_local_maxima(density_map, threshold=0.3, distance=10):
    """Findet die Koordinaten der Bienen in der Heatmap."""
    data_max = maximum_filter(density_map, footprint=np.ones((distance, distance)))
    maxima = (density_map == data_max)
    diff = (density_map > threshold)
    maxima &= diff
    return np.argwhere(maxima)

def run_prediction():
    # 1. Modell laden
    custom_dict = {'combined_loss': combined_loss, 'count_loss': count_loss}
    print("Lade Modell...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict)
    
    # 2. Bild vorbereiten
    img_raw = ImageOps.exif_transpose(Image.open(INPUT_IMAGE)).convert("RGB")
    scale = TARGET_DIM / max(img_raw.size)
    new_w, new_h = int(img_raw.size[0] * scale), int(img_raw.size[1] * scale)
    img_resized = img_raw.resize((new_w, new_h), Image.BILINEAR)
    
    # Arbeitskopie für Heatmap (wir sammeln die Vorhersagen)
    full_density_map = np.zeros((new_h, new_w))
    count_weights = np.zeros((new_h, new_w)) # Zum Mitteln der Überlappungen
    
    # 3. Sliding Window
    img_array = np.array(img_resized) / 255.0
    stride = TILE_SIZE // 2
    
    print("Analysiere Wabe...")
    for y in range(0, new_h - TILE_SIZE + 1, stride):
        for x in range(0, new_w - TILE_SIZE + 1, stride):
            tile = img_array[y:y+TILE_SIZE, x:x+TILE_SIZE]
            pred = model.predict(np.expand_dims(tile, axis=0), verbose=0)[0]
            
            # Heatmap ins Gesamtbild einfügen
            full_density_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += pred.squeeze()
            count_weights[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1

    # Normalisieren (Mittelwert bei Überlappung)
    full_density_map /= np.maximum(count_weights, 1)

    # 4. Bienen lokalisieren
    bee_coords = find_local_maxima(full_density_map, threshold=0.045, distance=20)
    bee_count = len(bee_coords)
    
    # Füge das nach der Zeile 'bee_count = len(bee_coords)' ein:
    print(f"DEBUG: Max-Wert in der Heatmap: {np.max(full_density_map)}")
    print(f"DEBUG: Summe der gesamten Heatmap: {np.sum(full_density_map)}")

    # 5. Zeichnen
    draw = ImageDraw.Draw(img_resized)
    for coord in bee_coords:
        y, x = coord
        draw.ellipse([x-3, y-3, x+3, y+3], fill='red', outline='white')
    
    # Text einfügen
    text = f"Gezählte Bienen: {bee_count}"
    print(f"--- {text} ---")
    draw.text((20, 20), text, fill="yellow") # Falls vorhanden: font=ImageFont.truetype(...)

    # Erstelle ein farbiges "Glühen" (Heatmap)
    heatmap_img = Image.fromarray((full_density_map * 255 * 5).clip(0, 255).astype(np.uint8))
    heatmap_img = heatmap_img.convert("L")
    heatmap_colored = ImageOps.colorize(heatmap_img, black="black", white="yellow", mid="red")
    
    # Lege die Heatmap halbtransparent über das Originalbild
    final_overlay = Image.blend(img_resized, heatmap_colored.convert("RGB"), alpha=0.4)
    final_overlay.save("heatmap_analyse.jpg")
    print("Analyse-Bild 'heatmap_analyse.jpg' wurde erstellt.")

    img_resized.save(OUTPUT_IMAGE)
    print(f"Ergebnis gespeichert in {OUTPUT_IMAGE}")

if __name__ == "__main__":
    run_prediction()