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
THRESHOLD = 0.012
DISTANCE = 28

# Stride (Überlappung): je kleiner, desto genauer aber langsamer
STRIDE = 200

def find_local_maxima(density_map, threshold=THRESHOLD, distance=DISTANCE):
    data_max = maximum_filter(density_map, footprint=np.ones((distance, distance)))
    maxima = (density_map == data_max)
    diff = (density_map > threshold)
    maxima &= diff
    return np.argwhere(maxima)

def _make_positions(length, tile_size, stride):
    """
    Erzeugt Startpositionen für Sliding Window und stellt sicher,
    dass der Rand (length - tile_size) immer enthalten ist.
    """
    if length <= tile_size:
        return [0]

    pos = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if pos[-1] != last:
        pos.append(last)
    return pos

def process_image(img_path, model, batch_size=32):
    # 1. Bild laden und EXIF-Rotation fixen
    img_raw = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    orig_w, orig_h = img_raw.size

    # 2. Skalierung auf 2000px (Synchron zum Training)
    max_dim = 2000
    scale = max_dim / max(orig_w, orig_h) if max(orig_w, orig_h) > max_dim else 1.0
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img_raw.resize((new_w, new_h), Image.BILINEAR)

    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    h, w, _ = img_array.shape

    # 3. Kacheln extrahieren – WICHTIG: Ränder IMMER abdecken!
    xs = _make_positions(w, TILE_SIZE, STRIDE)
    ys = _make_positions(h, TILE_SIZE, STRIDE)

    tiles = []
    coords = []

    for y in ys:
        for x in xs:
            tile = img_array[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Sicherheit: falls Bild kleiner als TILE_SIZE (selten), padde auf 256
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1], :] = tile
                tile = padded

            tiles.append(tile)
            coords.append((x, y))

    if not tiles:
        # sollte praktisch nicht passieren, aber sicher ist sicher
        return img_resized, 0.0, 0

    # 4. Batch-Vorhersage
    tiles = np.asarray(tiles, dtype=np.float32)
    predictions = model.predict(tiles, batch_size=batch_size, verbose=0)

    # 5. Dichtekarte zusammensetzen (Mittelwert in Überlappungen)
    density_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        pred_tile = predictions[i, :, :, 0]

        # Wenn wir gepadded haben (bei kleinen Bildern), beschneiden wir zurück
        y_end = min(y + TILE_SIZE, h)
        x_end = min(x + TILE_SIZE, w)
        ph = y_end - y
        pw = x_end - x

        density_map[y:y_end, x:x_end] += pred_tile[:ph, :pw]
        count_map[y:y_end, x:x_end] += 1.0

    density_map = np.divide(
        density_map, count_map,
        out=np.zeros_like(density_map),
        where=count_map != 0
    )

    # --- FINALE FEINJUSTIERUNG ---
    smoothed_map = gaussian_filter(density_map, sigma=1.55)

    # Mathematische Summe (die “Count”-Schätzung)
    math_sum = float(np.sum(density_map))

    CALIBRATION_FACTOR = 1.35
    math_sum_calibrated = math_sum * CALIBRATION_FACTOR


    # Peak-Zählung (optional / visuell)
    points = find_local_maxima(smoothed_map, threshold=THRESHOLD, distance=DISTANCE)
    bee_count = int(len(points))

    # Ergebnisbild: Punkte einzeichnen
    draw = ImageDraw.Draw(img_resized)
    for p in points:
        ry, rx = p
        draw.ellipse([rx-4, ry-4, rx+4, ry+4], fill='red', outline='white')

    return img_resized, math_sum_calibrated, bee_count

def main():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    custom_dict = {'combined_loss': combined_loss, 'count_loss': count_loss}
    print("Lade Modell...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict, compile=False)

    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Gefundene Bilder: {len(files)}")

    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dateiname', 'Mathe_Summe'])

        for filename in files:
            img_path = os.path.join(SOURCE_FOLDER, filename)
            try:
                res_img, m_sum, b_count = process_image(img_path, model)

                res_img.save(os.path.join(RESULT_FOLDER, f"result_{filename}"))
                writer.writerow([filename, round(m_sum, 2)])
            except Exception as e:
                print(f"Fehler bei {filename}: {e}")

    print(f"\nFertig! Statistiken in {CSV_FILE} gespeichert.")

if __name__ == "__main__":
    main()
