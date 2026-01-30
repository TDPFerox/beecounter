import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter
from model import combined_loss, count_loss

# --- KONFIGURATION ---
MODEL_PATH = 'Model/best_model.keras'
SOURCE_FOLDER = 'Data/testbilder'
RESULT_FOLDER = 'Data/ergebnisse'
CSV_FILE = 'Metric/statistik_zaehlung.csv'

TILE_SIZE = 256
THRESHOLD = 0.012  # Leicht gesenkt, da das Modell vorsichtiger ist
DISTANCE = 28     # Pixel-Abstand zwischen zwei Bienen-Maxima

def find_local_maxima(density_map, threshold=THRESHOLD, distance=DISTANCE):
    data_max = maximum_filter(density_map, footprint=np.ones((distance, distance)))
    maxima = (density_map == data_max)
    diff = (density_map > threshold)
    maxima &= diff
    return np.argwhere(maxima)

def process_image(img_path, model, batch_size=32):
    # 1. Bild laden und EXIF-Rotation fixen
    img_raw = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    orig_w, orig_h = img_raw.size
    
    # 2. SKALIERUNG AUF 2000PX (Synchron zum Training!)
    max_dim = 2000
    scale = max_dim / max(orig_w, orig_h) if max(orig_w, orig_h) > max_dim else 1.0
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img_raw.resize((new_w, new_h), Image.BILINEAR)
    
    img_array = np.array(img_resized) / 255.0
    h, w, _ = img_array.shape
    
    density_map = np.zeros((h, w), dtype=np.float32)
    
    # 3. Kacheln extrahieren
    tiles = []
    coords = []
    stride = 200 
    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            tile = img_array[y:y+TILE_SIZE, x:x+TILE_SIZE]
            tiles.append(tile)
            coords.append((x, y))
            
    # 4. Batch-Vorhersage
    tiles = np.array(tiles)
    predictions = model.predict(tiles, batch_size=batch_size, verbose=0)
    
    # 5. Dichtekarte zusammensetzen
    count_map = np.zeros((h, w), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        pred_tile = predictions[i, :, :, 0]
        density_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += pred_tile
        count_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1.0
        
    density_map = np.divide(density_map, count_map, out=np.zeros_like(density_map), where=count_map!=0)
    
    # --- FINALE FEINJUSTIERUNG ---
    
    # 1. Glättung bleibt bei 1.1 (guter Kompromiss)
    smoothed_map = gaussian_filter(density_map, sigma=1.55)
    
    # 2. KEIN fester Faktor mehr! Wir nehmen die Original-Dichte
    math_sum = np.sum(density_map) 
    
    # 3. Intelligente Punktsuche
    # Wir nehmen einen sehr niedrigen Threshold, aber eine größere DISTANCE
    # DISTANCE=25 verhindert, dass Kopf und Hinterleib separat gezählt werden.
    # Das wird IMG_3579 wieder Richtung 450 drücken.
    points = find_local_maxima(smoothed_map, threshold=THRESHOLD, distance=DISTANCE)
    bee_count = len(points)
    
    # 7. Ergebnis-Bild erstellen
    draw = ImageDraw.Draw(img_resized)
    for p in points:
        ry, rx = p
        draw.ellipse([rx-4, ry-4, rx+4, ry+4], fill='red', outline='white')
        
    return img_resized, math_sum, bee_count


def main():
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    if not os.path.exists(os.path.dirname(CSV_FILE)):
        os.makedirs(os.path.dirname(CSV_FILE))
        
    custom_dict = {'combined_loss': combined_loss, 'count_loss': count_loss}
    print("Lade Modell...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict, compile=False)
    
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Gefundene Bilder: {len(files)}")
    
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dateiname', 'Mathe_Summe', 'Gezaehlte_Bienen'])
        
        for filename in files:
            img_path = os.path.join(SOURCE_FOLDER, filename)
            try:
                res_img, m_sum, b_count = process_image(img_path, model)
                print(f"-> {filename}: Summe={m_sum:.1f}, Punkte={b_count}")
                
                res_img.save(os.path.join(RESULT_FOLDER, f"result_{filename}"))
                writer.writerow([filename, round(m_sum, 2), b_count])
            except Exception as e:
                print(f"Fehler bei {filename}: {e}")

    print(f"\nFertig! Statistiken in {CSV_FILE} gespeichert.")

if __name__ == "__main__":
    main()