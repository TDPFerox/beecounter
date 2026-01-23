import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label
import os
from scipy.ndimage import gaussian_filter

# 1. Spezielle Loss-Funktionen (wie gehabt)
def count_loss(y_true, y_pred):
    return tf.abs(tf.reduce_sum(y_true) - tf.reduce_sum(y_pred))

def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return mse + 0.1 * count_loss(y_true, y_pred)

# Modell laden
print("Lade Modell...")
model = tf.keras.models.load_model('best_model.keras', 
    custom_objects={'count_loss': count_loss, 'combined_loss': combined_loss})

from scipy.ndimage import gaussian_filter, maximum_filter, label

def get_coordinates(prediction, threshold=0.15):
    """Extrahiert Punkt-Koordinaten mit robuster Gipfel-Findung."""
    
    # 1. Sehr sanfte Glättung nur für die Suche nach dem 'Hügel-Gipfel'
    # Wir nehmen sigma=0.5, das ist fast gar nichts, aber hilft gegen Pixel-Rauschen
    smoothed = gaussian_filter(prediction, sigma=4)
    
    # 2. Lokale Maxima finden (7x7 Fenster ist meist ideal für Bienen)
    data_max = maximum_filter(smoothed, size=7)
    maxima = (smoothed == data_max)
    
    # 3. WICHTIG: Wir prüfen den Threshold gegen die ORIGINAL-Werte
    # So verhindern wir, dass die Glättung uns die Bienen "löscht"
    significant = (prediction > threshold)
    maxima &= significant
    
    # Koordinaten extrahieren
    labeled, num_objects = label(maxima)
    coords = []
    for i in range(1, num_objects + 1):
        where = np.where(labeled == i)
        if len(where[0]) > 0:
            # Mittelpunkt der gefundenen Koordinate
            y, x = np.mean(where[0]), np.mean(where[1])
            coords.append((x, y))
    return coords

def run_prediction(image_path):
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    
    # 1. Skalierung und 16er-Raster (wie gehabt)
    max_dim = 2000
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    final_w, final_h = (new_w // 16) * 16, (new_h // 16) * 16
    
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    img_cropped = img_resized.crop((0, 0, final_w, final_h))
    
    # 2. Vorhersage
    img_array = np.array(img_cropped) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    print(f"Verarbeite Bild: {image_path}...")
    prediction = model.predict(img_input)[0, :, :, 0]
    
    bee_count = np.sum(prediction)
    coords = get_coordinates(prediction, threshold=0.03) # Schwellenwert für Punkte
    
    # 3. Visualisierung: Heatmap-Overlay
    vis_prediction = np.copy(prediction)
    vis_prediction[vis_prediction < 0.02] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    
    # Linke Seite: Sauberes Heatmap Overlay
    ax[0].imshow(img_cropped)
    ax[0].imshow(vis_prediction, cmap='hot', alpha=0.6)
    ax[0].set_title(f"KI Heatmap (Gezählt: {int(round(bee_count))})")
    ax[0].axis('off')
    
    # Rechte Seite: Original mit Kreisen
    ax[1].imshow(img_cropped)
    for x, y in coords:
        circle = plt.Circle((x, y), radius=6, color='red', fill=False, linewidth=1.5)
        ax[1].add_patch(circle)
    
    ax[1].set_title(f"KI Einzelerkennung (Gefunden: {len(coords)} Punkte)")
    ax[1].axis('off')
    
    plt.tight_layout()
    output_name = "detection_check_" + os.path.basename(image_path)
    plt.savefig(output_name, dpi=300)
    print(f"Ergebnis gespeichert als: {output_name}")
    print(f"Summen-Zählung (Dichte): {int(round(bee_count))}")
    print(f"Punkt-Zählung (Kreise): {len(coords)}")
    plt.show()

if __name__ == "__main__":
    test_bild = "Wabenbilder/IMG_3579.jpg" 
    if os.path.exists(test_bild):
        run_prediction(test_bild)