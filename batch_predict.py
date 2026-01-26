import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import maximum_filter, gaussian_filter
from model import combined_loss, count_loss

# --- KONFIGURATION ---
MODEL_PATH = 'best_model.keras'
SOURCE_FOLDER = 'testbilder'
RESULT_FOLDER = 'ergebnisse'
CSV_FILE = 'statistik_zaehlung.csv'

TARGET_DIM = 2000
TILE_SIZE = 256
# Deine kalibrierten Werte:
THRESHOLD = 0.025
DISTANCE = 15

def find_local_maxima(density_map, threshold=0.025, distance=15):
    data_max = maximum_filter(density_map, footprint=np.ones((distance, distance)))
    maxima = (density_map == data_max)
    diff = (density_map > threshold)
    maxima &= diff
    return np.argwhere(maxima)

def process_image(img_path, model):
    # Bild laden
    img_raw = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    scale = TARGET_DIM / max(img_raw.size)
    new_w, new_h = int(img_raw.size[0] * scale), int(img_raw.size[1] * scale)
    img_resized = img_raw.resize((new_w, new_h), Image.BILINEAR)
    
    full_density_map = np.zeros((new_h, new_w))
    count_weights = np.zeros((new_h, new_w))
    img_array = np.array(img_resized) / 255.0
    stride = TILE_SIZE // 2
    
    # Vorhersage-Schleife
    for y in range(0, new_h - TILE_SIZE + 1, stride):
        for x in range(0, new_w - TILE_SIZE + 1, stride):
            tile = img_array[y:y+TILE_SIZE, x:x+TILE_SIZE]
            pred = model.predict(np.expand_dims(tile, axis=0), verbose=0)[0]
            full_density_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += pred.squeeze()
            count_weights[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1

    full_density_map /= np.maximum(count_weights, 1)
    full_density_map = gaussian_filter(full_density_map, sigma=0.8) # Glättung für Stabilität
    
    math_sum = np.sum(full_density_map)
    bee_coords = find_local_maxima(full_density_map, threshold=THRESHOLD, distance=DISTANCE)
    bee_count = len(bee_coords)
    
    # Ergebnisbild zeichnen
    draw = ImageDraw.Draw(img_resized)
    for coord in bee_coords:
        y, x = coord
        draw.ellipse([x-3, y-3, x+3, y+3], fill='red', outline='white')
    
    return img_resized, math_sum, bee_count

def main():
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
        
    custom_dict = {'combined_loss': combined_loss, 'count_loss': count_loss}
    print("Lade Modell...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict)
    
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Gefundene Bilder: {len(files)}")
    
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dateiname', 'Mathe_Summe', 'Gezaehlte_Bienen', 'Differenz_Prozent'])
        
        for filename in files:
            print(f"Verarbeite {filename}...")
            img_path = os.path.join(SOURCE_FOLDER, filename)
            res_img, m_sum, b_count = process_image(img_path, model)
            
            # Prozentualen Unterschied berechnen
            # (Gezählt - Mathe_Summe) / Mathe_Summe * 100
            if m_sum > 0:
                diff_pct = ((b_count - m_sum) / m_sum) * 100
            else:
                diff_pct = 0
            
            # Speichern
            res_img.save(os.path.join(RESULT_FOLDER, f"result_{filename}"))
            
            # In CSV schreiben
            writer.writerow([
                filename, 
                round(m_sum, 2), 
                b_count, 
                f"{round(diff_pct, 2)}%"
            ])
            
            print(f"  -> Gefunden: {b_count} (Summe: {round(m_sum, 1)} | Abweichung: {round(diff_pct, 1)}%)")

    print(f"\nFertig! Statistiken in {CSV_FILE} gespeichert.")

if __name__ == "__main__":
    main()