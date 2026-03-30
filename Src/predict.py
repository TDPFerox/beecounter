import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import maximum_filter
from model import count_loss, weighted_total_loss

# --- KONFIGURATION ---
MODEL_PATH = 'Model/best_model.keras'
INPUT_IMAGE = '1a.JPG'
OUTPUT_IMAGE = 'resultat_zaehlung.jpg'
HEATMAP_OUTPUT_IMAGE = 'heatmap_analyse.jpg'
TARGET_DIM = 2000 
TILE_SIZE = 256
STRIDE = 128
BATCH_SIZE = 32

def find_local_maxima(density_map, threshold=0.3, distance=10):
    """Findet die Koordinaten der Bienen in der Heatmap."""
    data_max = maximum_filter(density_map, footprint=np.ones((distance, distance)))
    maxima = (density_map == data_max)
    diff = (density_map > threshold)
    maxima &= diff
    return np.argwhere(maxima)


def _make_positions(length, tile_size, stride):
    """Erzeugt Sliding-Window-Startpositionen inkl. letzter Randposition."""
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def build_density_map(model, image_path, batch_size=BATCH_SIZE):
    """Lädt ein Bild und erstellt daraus eine vollständige Dichtekarte."""
    img_raw = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    orig_w, orig_h = img_raw.size

    scale = TARGET_DIM / max(orig_w, orig_h) if max(orig_w, orig_h) > TARGET_DIM else 1.0
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img_raw.resize((new_w, new_h), Image.Resampling.BILINEAR)

    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
    h, w, _ = img_array.shape

    xs = _make_positions(w, TILE_SIZE, STRIDE)
    ys = _make_positions(h, TILE_SIZE, STRIDE)

    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            tile = img_array[y:y + TILE_SIZE, x:x + TILE_SIZE]

            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1], :] = tile
                tile = padded

            tiles.append(tile)
            coords.append((x, y))

    if not tiles:
        return img_resized, np.zeros((h, w), dtype=np.float32)

    predictions = model.predict(np.asarray(tiles, dtype=np.float32), batch_size=batch_size, verbose=0)

    density_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        pred_tile = predictions[i, :, :, 0]
        y_end = min(y + TILE_SIZE, h)
        x_end = min(x + TILE_SIZE, w)
        ph = y_end - y
        pw = x_end - x

        density_map[y:y_end, x:x_end] += pred_tile[:ph, :pw]
        count_map[y:y_end, x:x_end] += 1.0

    density_map = np.divide(
        density_map,
        count_map,
        out=np.zeros_like(density_map),
        where=count_map != 0,
    )

    return img_resized, density_map

def run_prediction():
    # 1. Modell laden
    custom_dict = {'weighted_total_loss': weighted_total_loss, 'count_loss': count_loss}
    print("Lade Modell...")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict, compile=False)

    print("Analysiere Wabe...")
    img_resized, full_density_map = build_density_map(model, INPUT_IMAGE)

    # 4. Bienen lokalisieren
    bee_coords = find_local_maxima(full_density_map, threshold=0.045, distance=20)
    bee_count = len(bee_coords)

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
    final_overlay.save(HEATMAP_OUTPUT_IMAGE)
    print(f"Analyse-Bild '{HEATMAP_OUTPUT_IMAGE}' wurde erstellt.")

    img_resized.save(OUTPUT_IMAGE)
    print(f"Ergebnis gespeichert in {OUTPUT_IMAGE}")

if __name__ == "__main__":
    run_prediction()