import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import os
import glob
from prepare_data import parse_annotations
from densitymap import generate_density_map

def analyze_density_map(dens_map):
    
    # Summe berechnen (Das ist das, was die KI lernen soll)
    total_bees = np.sum(dens_map)
    max_val = np.max(dens_map)

    print(f"Summe der Bienen in dieser Karte: {total_bees:.2f}")
    print(f"Höchster Pixelwert: {max_val:.6f}")

    plt.figure(figsize=(12, 5))

    # 1. Die Karte visualisieren
    plt.subplot(1, 2, 1)
    plt.imshow(dens_map, cmap='hot')
    plt.colorbar(label='Dichtewert')
    plt.title(f"Dichtekarte (Summe: {int(total_bees)})")

    # 2. Ein Profil-Schnitt durch ein dichtes Cluster
    # Wir nehmen die mittlere Zeile der Karte
    plt.subplot(1, 2, 2)
    mid_row = dens_map[dens_map.shape[0]//2, :]
    plt.plot(mid_row)
    plt.title("Querschnitt durch die Map")
    plt.xlabel("Pixel")
    plt.ylabel("Dichte")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Metric/density_map.png')

# TEST: Ersetze 'y_XXX.npy' mit dem Pfad zu einer Kachel von Bild 3a
# Du findest die Kacheln in Data/prepared_data/train/ oder val/
# Such dir eine aus, die nach viel "Action" aussieht.
# analyze_density_map('Data/prepared_data/train/y_0.npy')

if __name__ == "__main__":

    annotations = []
    xml_path = 'Data/annotations'
    images_folder = 'Data/images'
    mode = 'train'

    xml_files = glob.glob(os.path.join(xml_path, mode, '*.xml'))
    if not xml_files:
        xml_files = [xml_path] if os.path.isfile(xml_path) else []

    for f in xml_files: 
        annotations.extend(parse_annotations(f))

    tile_counter = 0
    print(f"--- Modus: {mode.upper()} ---")

    for ann in annotations:
        print(f"Suche nach 8b.jpg, gefunden: '{ann['image_name']}'")
        if ann['image_name'] == '8b.JPG':
            print("Image found")
            img_path = os.path.join(images_folder, mode, ann['image_name'])
            if not os.path.exists(img_path): continue
            
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            
            # Skalierung auf 2000px
            max_dim = 2000
            w, h = img.size
            scale = max_dim / max(w, h) if max(w, h) > max_dim else 1.0
            if scale != 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            
            points = [(p[0] * scale, p[1] * scale) for p in ann['points']]
            # Wir behalten das Original-Bild-Array für die Augmentation
            img_np_orig = np.array(img) 
            full_density = generate_density_map(points, img.height, img.width, sigma=2.5)
            analyze_density_map(full_density)
        else:
            continue
    