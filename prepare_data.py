import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from densitymap import generate_density_map

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for image in root.findall('image'):
        image_name = image.get('name')
        points = []
        for point in image.findall('points'):
            if point.get('label') == 'Biene':
                coords = point.get('points').split(',')
                points.append((float(coords[0]), float(coords[1])))
        
        # NEU: Wir nehmen das Bild IMMER auf, auch wenn points leer ist!
        annotations.append({
            'image_name': image_name, 
            'width': int(image.get('width')),
            'height': int(image.get('height')), 
            'points': points
        })
    return annotations

def get_tiles_and_augment(img, points, tile_size=256, stride=128):
    w, h = img.size
    img_np = np.array(img, dtype=np.float32) / 255.0
    full_density = generate_density_map(points, h, w, sigma=4)
    
    tx, ty = [], []
    # Wir loopen durch das Bild
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_img = img_np[y:y+tile_size, x:x+tile_size]
            tile_dens = full_density[y:y+tile_size, x:x+tile_size]
            
            # Nur Kacheln mit Inhalt (Summe > 0.05 simuliert eine halbe Biene)
            if np.sum(tile_dens) > 0.05:
                tx.append(tile_img)
                ty.append(tile_dens)
                # Augmentation: Nur horizontal spiegeln spart 50% RAM gegenüber h+v
                tx.append(np.flip(tile_img, axis=1))
                ty.append(np.flip(tile_dens, axis=1))
    return tx, ty

def prepare_training_data(xml_path, images_folder, output_folder='prepared_data', tile_size=256):
    # Unterordner für die einzelnen Kacheln erstellen
    tiles_dir = os.path.join(output_folder, 'tiles')
    if not os.path.exists(tiles_dir): 
        os.makedirs(tiles_dir)
    
    annotations = []
    xml_files = glob.glob(os.path.join(xml_path, '*.xml')) if os.path.isdir(xml_path) else [xml_path]
    for f in xml_files: 
        annotations.extend(parse_annotations(f))

    tile_counter = 0
    print(f"Starte RAM-schonendes Tiling mit 8-fach Augmentation für {len(annotations)} Bilder...")

    for ann in annotations:
        img_path = os.path.join(images_folder, ann['image_name'])
        if not os.path.exists(img_path): 
            continue
        
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        
        # Sicherheits-Skalierung auf max 2000px (wie in deinem aktuellen Setup)
        max_dim = 2000
        w, h = img.size
        scale = 1.0
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        
        points = [(p[0] * scale, p[1] * scale) for p in ann['points']]
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Erzeuge die Dichtekarte (Sigma 8 für bessere Lernbarkeit bei 2000px)
        full_density = generate_density_map(points, img.height, img.width, sigma=4)

        # Tiling mit Überlappung (Stride 128 bei 256er Kacheln)
        stride = 128
        for y in range(0, img.height - tile_size + 1, stride):
            for x in range(0, img.width - tile_size + 1, stride):
                tile_img = img_np[y:y+tile_size, x:x+tile_size]
                tile_dens = full_density[y:y+tile_size, x:x+tile_size]
                
                has_bees = np.sum(tile_dens) > 0.05

                # Nur Kacheln mit Inhalt verarbeiten
                if has_bees or (np.random.random() < 0.05):
                    for k in range(4):
                        rot_img = np.rot90(tile_img, k=k)
                        rot_dens = np.rot90(tile_dens, k=k)
                        
                        # 1. Rotierte Version speichern
                        np.save(os.path.join(tiles_dir, f"x_{tile_counter}.npy"), rot_img)
                        np.save(os.path.join(tiles_dir, f"y_{tile_counter}.npy"), rot_dens)
                        tile_counter += 1
                        
                        # 2. Zusätzlich gespiegelte Version der Rotation speichern (Horizontaler Flip)
                        np.save(os.path.join(tiles_dir, f"x_{tile_counter}.npy"), np.flip(rot_img, axis=1))
                        np.save(os.path.join(tiles_dir, f"y_{tile_counter}.npy"), np.flip(rot_dens, axis=1))
                        tile_counter += 1
        
        print(f"  - {ann['image_name']} verarbeitet. Kacheln gesamt: {tile_counter}")

    print(f"\n✓ Fertig! {tile_counter} Kacheln (inkl. 8-fach Augmentation) in {tiles_dir} gespeichert.")
    return tile_counter