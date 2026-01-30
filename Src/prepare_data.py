import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageEnhance
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
        
        annotations.append({
            'image_name': image_name, 
            'width': int(image.get('width')),
            'height': int(image.get('height')), 
            'points': points
        })
    return annotations

def prepare_training_data(xml_path, images_folder, output_folder, mode='train', tile_size=256):
    target_dir = os.path.join(output_folder, mode)
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    
    annotations = []
    xml_files = glob.glob(os.path.join(xml_path, mode, '*.xml'))
    if not xml_files:
        xml_files = [xml_path] if os.path.isfile(xml_path) else []

    for f in xml_files: 
        annotations.extend(parse_annotations(f))

    tile_counter = 0
    print(f"--- Modus: {mode.upper()} ---")
    
    for ann in annotations:
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

        is_empty = len(points) == 0
        # Kleinerer Stride bei vollen Bildern im Training, um mehr Kacheln zu generieren
        stride = 64 if (len(points) > 400 and mode == 'train') else 128
        empty_prob = 1.0 if (is_empty and mode == 'train') else 0.05

        for y in range(0, img.height - tile_size + 1, stride):
            for x in range(0, img.width - tile_size + 1, stride):
                tile_img_raw = img_np_orig[y:y+tile_size, x:x+tile_size]
                tile_dens = full_density[y:y+tile_size, x:x+tile_size]
                
                bee_count_in_tile = np.sum(tile_dens)
                has_bees = bee_count_in_tile > 0.05

                if has_bees or (np.random.random() < empty_prob):
                    # --- NEU: MULTIPLIER LOGIK ---
                    if mode == 'train':
                        if bee_count_in_tile > 15:    multiplier = 10 # Sehr dichte Kacheln 10x
                        elif bee_count_in_tile > 5:   multiplier = 3  # Mittlere Dichte 3x
                        else:                         multiplier = 1  # Wenig/keine Bienen 1x
                    else:
                        multiplier = 1 # Validierung wird nicht künstlich aufgebläht

                    for _ in range(multiplier):
                        # 1. Helligkeits-Augmentierung (PIL)
                        t_img_pil = Image.fromarray(tile_img_raw)
                        
                        if mode == 'train':
                            # Zufällige Helligkeit 0.7 - 1.3
                            t_img_pil = ImageEnhance.Brightness(t_img_pil).enhance(np.random.uniform(0.7, 1.3))
                            # Zufälliger Kontrast 0.8 - 1.2
                            t_img_pil = ImageEnhance.Contrast(t_img_pil).enhance(np.random.uniform(0.8, 1.2))
                        
                        aug_img = np.array(t_img_pil).astype(np.float32) / 255.0
                        aug_dens = tile_dens.copy()

                        # 2. Geometrische Augmentierung (Zufällig statt starr)
                        if mode == 'train':
                            k = np.random.randint(0, 4)
                            aug_img = np.rot90(aug_img, k=k)
                            aug_dens = np.rot90(aug_dens, k=k)
                            if np.random.random() > 0.5:
                                aug_img = np.flip(aug_img, axis=1)
                                aug_dens = np.flip(aug_dens, axis=1)
                        
                        # Speichern
                        np.save(os.path.join(target_dir, f"x_{tile_counter}.npy"), aug_img)
                        np.save(os.path.join(target_dir, f"y_{tile_counter}.npy"), aug_dens)
                        tile_counter += 1
        
        print(f"  - {ann['image_name']} fertig. Kacheln Stand: {tile_counter}")

    return tile_counter