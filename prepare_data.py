import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from densitymap import generate_density_map

def parse_annotations(xml_path):
    """
    Parst die XML-Annotationsdatei und extrahiert Bildnamen und Bienenpositionen.
    
    Returns:
        List von Dictionaries mit 'image_name', 'width', 'height', und 'points'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    
    for image in root.findall('image'):
        image_name = image.get('name')
        orig_width = int(image.get('width'))
        orig_height = int(image.get('height'))
        
        points = []
        for point in image.findall('points'):
            if point.get('label') == 'Biene':
                coords = point.get('points').split(',')
                x = float(coords[0])
                y = float(coords[1])
                points.append((x, y))
        
        if points:  # Nur Bilder mit Annotationen hinzufügen
            annotations.append({
                'image_name': image_name,
                'width': orig_width,
                'height': orig_height,
                'points': points
            })
    
    return annotations


def resize_image_and_adjust_points(image_path, points, orig_width, orig_height, 
                                   target_width=512, target_height=288):
    """
    Lädt und skaliert ein Bild auf die Zielgröße und passt die Annotationspunkte an.
    Berücksichtigt EXIF-Orientierung und dreht nur Hochkant-Bilder.
    
    Returns:
        resized_image (PIL Image), adjusted_points (list of tuples)
    """
    img = Image.open(image_path)
    
    # WICHTIG: Korrigiere zuerst die EXIF-Orientierung
    # (Kameras/Handys speichern oft die Orientierung in EXIF-Metadaten)
    img = ImageOps.exif_transpose(img)
    if img is None:
        img = Image.open(image_path)
    img = img.convert("RGB")
    
    # Hole die tatsächliche Bildgröße NACH EXIF-Korrektur
    actual_width, actual_height = img.size
    
    # Prüfe ob Bild TATSÄCHLICH hochkant ist (nach EXIF-Korrektur)
    if actual_height > actual_width:
        print(f"  → Drehe Hochkant-Bild um 90° (von {actual_width}x{actual_height} zu {actual_height}x{actual_width})")
        # Drehe Bild um 90° gegen Uhrzeigersinn (Hochkant → Querformat)
        img = img.transpose(Image.ROTATE_90)
        
        # Berechne Skalierungsfaktor von XML-Koordinaten zu tatsächlichen Bild-Koordinaten
        scale_x = actual_width / orig_width
        scale_y = actual_height / orig_height
        
        # Passe Koordinaten an die Drehung an
        # Zuerst auf tatsächliche Bildkoordinaten skalieren
        # Bei 90° gegen UZS: new_x = y, new_y = actual_width - x
        rotated_points = []
        for x, y in points:
            # Skaliere auf tatsächliche Bildkoordinaten
            actual_x = x * scale_x
            actual_y = y * scale_y
            # Drehe
            new_x = actual_y
            new_y = actual_width - actual_x
            rotated_points.append((new_x, new_y))
        points = rotated_points
        
        # Tausche Dimensionen nach Drehung
        actual_width, actual_height = actual_height, actual_width
    else:
        # Keine Drehung nötig, aber Koordinaten müssen auf tatsächliche Bildgröße skaliert werden
        scale_x = actual_width / orig_width
        scale_y = actual_height / orig_height
        scaled_points = []
        for x, y in points:
            scaled_points.append((x * scale_x, y * scale_y))
        points = scaled_points
    
    # Jetzt sind Bild und Punkte in der gleichen Größe (actual_width x actual_height)
    # Berechne Skalierung um Aspect Ratio beizubehalten
    scale = min(target_width / actual_width, target_height / actual_height)
    new_w = int(actual_width * scale)
    new_h = int(actual_height * scale)
    
    # Resize mit beibehaltener Ratio
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Erstelle neues Bild mit schwarzem Hintergrund
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    
    # Zentriere das resized Bild
    left = (target_width - new_w) // 2
    top = (target_height - new_h) // 2
    
    new_img.paste(resized, (left, top))
    
    # Passe die Punkte an
    adjusted_points = []
    for x, y in points:
        # Skaliere die Koordinaten
        new_x = x * scale + left
        new_y = y * scale + top
        adjusted_points.append((new_x, new_y))
    
    return new_img, adjusted_points


def prepare_training_data(xml_path, images_folder, output_folder='prepared_data', 
                          target_width=512, target_height=288, sigma=4):
    """
    Hauptfunktion zur Vorbereitung der Trainingsdaten.
    
    Args:
        xml_path: Pfad zur annotations.xml ODER Ordner mit mehreren .xml Dateien
        images_folder: Ordner mit den Originalbildern
        output_folder: Ordner für die vorbereiteten Daten
        target_width, target_height: Zielgröße der Bilder
        sigma: Sigma-Wert für den Gaussian-Filter der Dichtekarte
    """
    # Erstelle Ausgabeordner
    os.makedirs(output_folder, exist_ok=True)
    images_output = os.path.join(output_folder, 'images')
    density_output = os.path.join(output_folder, 'density_maps')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(density_output, exist_ok=True)
    
    # Parse Annotationen (einzelne Datei oder alle XML-Dateien im Ordner)
    print("Parse Annotationen...")
    annotations = []
    
    if os.path.isdir(xml_path):
        # Suche alle .xml Dateien im Ordner
        xml_files = glob.glob(os.path.join(xml_path, '*.xml'))
        print(f"Gefundene XML-Dateien: {len(xml_files)}")
        for xml_file in xml_files:
            print(f"  - Lese {os.path.basename(xml_file)}")
            annotations.extend(parse_annotations(xml_file))
    else:
        # Einzelne XML-Datei
        annotations = parse_annotations(xml_path)
    
    print(f"Gesamt: {len(annotations)} annotierte Bilder")
    
    X_images = []
    Y_density_maps = []
    processed_names = []
    
    for i, ann in enumerate(annotations):
        image_name = ann['image_name']
        image_path = os.path.join(images_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warnung: Bild {image_name} nicht gefunden, überspringe...")
            continue
        
        print(f"Verarbeite {i+1}/{len(annotations)}: {image_name} ({len(ann['points'])} Bienen)")
        
        # Lade und skaliere Bild, passe Punkte an
        resized_img, adjusted_points = resize_image_and_adjust_points(
            image_path, ann['points'], ann['width'], ann['height'],
            target_width, target_height
        )
        
        # Generiere Dichtekarte
        density_map = generate_density_map(
            adjusted_points, target_height, target_width, sigma=sigma
        )
        
        # Konvertiere Bild zu NumPy Array
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        
        # Speichere als .npy Dateien
        base_name = os.path.splitext(image_name)[0]
        np.save(os.path.join(images_output, f"{base_name}.npy"), img_array)
        np.save(os.path.join(density_output, f"{base_name}.npy"), density_map)
        
        X_images.append(img_array)
        Y_density_maps.append(density_map)
        processed_names.append(image_name)
        
        # Optional: Speichere auch als Bild zur Visualisierung
        resized_img.save(os.path.join(images_output, f"{base_name}.jpg"))
    
    # Konvertiere zu NumPy Arrays
    X_images = np.array(X_images, dtype=np.float32)
    Y_density_maps = np.array(Y_density_maps, dtype=np.float32)
    Y_density_maps = np.expand_dims(Y_density_maps, axis=-1)  # Füge Channel-Dimension hinzu
    
    # Speichere als einzelne Dateien
    np.save(os.path.join(output_folder, 'X_train.npy'), X_images)
    np.save(os.path.join(output_folder, 'Y_train.npy'), Y_density_maps)
    
    print(f"\nDatenvorbereitung abgeschlossen!")
    print(f"X_train Shape: {X_images.shape}")
    print(f"Y_train Shape: {Y_density_maps.shape}")
    print(f"Gespeichert in: {output_folder}")
    
    # Speichere Statistiken
    with open(os.path.join(output_folder, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Anzahl Bilder: {len(X_images)}\n")
        f.write(f"Bildgröße: {target_width}x{target_height}\n")
        f.write(f"Sigma für Dichtekarte: {sigma}\n")
        f.write(f"\nVerarbeitete Bilder:\n")
        for i, name in enumerate(processed_names):
            total_bees = int(np.sum(Y_density_maps[i]))
            f.write(f"  {name}: ~{total_bees} Bienen\n")
    
    return X_images, Y_density_maps


if __name__ == "__main__":
    # Pfade definieren
    xml_path = "test"  # Ordner mit allen XML-Dateien (oder Pfad zu einzelner .xml)
    images_folder = "Wabenbilder"
    output_folder = "prepared_data"
    
    # Datenvorbereitung durchführen
    X_train, Y_train = prepare_training_data(
        xml_path=xml_path,
        images_folder=images_folder,
        output_folder=output_folder,
        target_width=512,
        target_height=288,
        sigma=4
    )
    
    print("\nDaten sind bereit für das Training!")
    print("Verwende: X_train = np.load('prepared_data/X_train.npy')")
    print("          Y_train = np.load('prepared_data/Y_train.npy')")
