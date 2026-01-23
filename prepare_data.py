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
        if points:
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
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    # XMLs laden
    annotations = []
    xml_files = glob.glob(os.path.join(xml_path, '*.xml')) if os.path.isdir(xml_path) else [xml_path]
    for f in xml_files: annotations.extend(parse_annotations(f))

    X_list, Y_list = [], []
    print(f"Starte Tiling für {len(annotations)} Bilder...")

    for ann in annotations:
        img_path = os.path.join(images_folder, ann['image_name'])
        if not os.path.exists(img_path): continue
        
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        
        # --- NEU: Sicherheits-Skalierung auf max 2000px ---
        max_dim = 2000
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # Punkte müssen im gleichen Verhältnis skaliert werden!
            points = [(p[0] * scale, p[1] * scale) for p in ann['points']]
        else:
            points = ann['points']
        # ------------------------------------------------
        
        tx, ty = get_tiles_and_augment(img, points, tile_size=tile_size)
        X_list.extend(tx)
        Y_list.extend(ty)
        print(f"  - {ann['image_name']}: {len(tx)} Kacheln (skaliert).")

    print("Konvertiere in Arrays... (hier crasht es oft bei RAM-Mangel)")
    X = np.array(X_list, dtype=np.float32)
    Y = np.expand_dims(np.array(Y_list, dtype=np.float32), axis=-1)
    
    np.save(os.path.join(output_folder, 'X_train.npy'), X)
    np.save(os.path.join(output_folder, 'Y_train.npy'), Y)
    print(f"Erfolgreich! {len(X)} Kacheln gespeichert.")
    return X, Y