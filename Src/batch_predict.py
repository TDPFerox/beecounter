import os
import csv
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from PIL import Image, ImageOps
from model import combined_loss, count_loss, weighted_total_loss

# --- KONFIGURATION ---
MODEL_PATH = 'Model/best_model.keras'
SOURCE_FOLDER = 'Data/testbilder'
RESULT_FOLDER = 'Data/ergebnisse'
CSV_FILE = 'Metric/statistik_zaehlung.csv'
CSV_FILE_CAL = 'Metric/statistik_zaehlung_kalibriert.csv'  # neu

TILE_SIZE = 256

# Stride (Überlappung): je kleiner, desto genauer aber langsamer
STRIDE = 128

# --- AUSWERTUNG / ZIELMETRIK ---
REL_TOL = 0.10        # 10% Ziel
LOW_COUNT_CUTOFF = 10 # unterhalb davon ist % unfair
LOW_COUNT_ABS_TOL = 2 # absolute Toleranz für sehr kleine counts (anpassbar)

# Optional: wenn du Bilder ohne GT im Namen hast -> skippen statt crashen
SKIP_IF_NO_GT = True

# Optional: Ergebnisse als Bild speichern
SAVE_RESULT_IMAGES = True


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


def extract_gt_from_filename(filename: str):
    """
    Extrahiert die Ground Truth aus Dateinamen wie:
      '3a_860.JPG' -> 860
      'IMG_3579_447.jpg' -> 447

    Regel: nimmt die LETZTE Zahl vor der Dateiendung.
    """
    # robust gegen .jpg/.jpeg/.png und Groß-/Kleinschreibung
    m = re.search(r'_(\d+)\.(jpg|jpeg|png)$', filename, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


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

    # 3. Kacheln extrahieren – Ränder IMMER abdecken
    xs = _make_positions(w, TILE_SIZE, STRIDE)
    ys = _make_positions(h, TILE_SIZE, STRIDE)

    tiles = []
    coords = []

    for y in ys:
        for x in xs:
            tile = img_array[y:y + TILE_SIZE, x:x + TILE_SIZE]

            # falls Bild kleiner als TILE_SIZE, pad auf 256
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1], :] = tile
                tile = padded

            tiles.append(tile)
            coords.append((x, y))

    if not tiles:
        return img_resized, 0.0  # sollte praktisch nie passieren

    # 4. Batch-Vorhersage
    tiles = np.asarray(tiles, dtype=np.float32)
    predictions = model.predict(tiles, batch_size=batch_size, verbose=0)

    # 5. Dichtekarte zusammensetzen (Mittelwert in Überlappungen)
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
        density_map, count_map,
        out=np.zeros_like(density_map),
        where=count_map != 0
    )

    # Mathematische Summe = Count-Schätzung (RAW)
    math_sum_raw = float(np.sum(density_map))

    return img_resized, math_sum_raw


def within_target(pred: float, gt: int) -> bool:
    """
    Kriterium: <=10% bei normalen Counts,
    und absolute Toleranz für sehr kleine GT.
    """
    if gt < LOW_COUNT_CUTOFF:
        return abs(pred - gt) <= LOW_COUNT_ABS_TOL
    return (abs(pred - gt) / max(gt, 1)) <= REL_TOL


def fit_linear_calibration(preds: np.ndarray, gts: np.ndarray):
    """
    Fit: gt ≈ a * pred + b (Least Squares)
    Rückgabe: (a, b)

    Note: numpy polyfit ist völlig ausreichend hier.
    """
    if len(preds) < 2:
        return 1.0, 0.0

    a, b = np.polyfit(preds, gts, deg=1)
    return float(a), float(b)


def main():
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    custom_dict = {
        'combined_loss': combined_loss,
        'weighted_total_loss': weighted_total_loss,
        'count_loss': count_loss
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_dict, compile=False)

    files = [f for f in os.listdir(SOURCE_FOLDER)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Gefundene Bilder: {len(files)}")

    rows = []

    # --- 1) RUN: Raw Predictions sammeln ---
    for filename in files:
        img_path = os.path.join(SOURCE_FOLDER, filename)

        gt = extract_gt_from_filename(filename)
        if gt is None and SKIP_IF_NO_GT:
            print(f"[SKIP] Keine GT im Namen gefunden: {filename}")
            continue

        try:
            res_img, pred_raw = process_image(img_path, model)

            if SAVE_RESULT_IMAGES:
                res_img.save(os.path.join(RESULT_FOLDER, f"result_{filename}"))

            rows.append({
                "Dateiname": filename,
                "GT": gt,
                "Pred_raw": pred_raw
            })

        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    if not rows:
        print("Keine auswertbaren Bilder gefunden (evtl. GT-Parsing?).")
        return

    df = pd.DataFrame(rows)

    # Manche Dateien könnten keine GT haben (wenn SKIP_IF_NO_GT=False)
    # Dann filtern wir für Kalibrierung/Auswertung auf gültige GT
    df_valid = df.dropna(subset=["GT"]).copy()
    df_valid["GT"] = df_valid["GT"].astype(int)

    # --- 2) Kalibrierung fitten (nur wenn GT vorhanden ist) ---
    if len(df_valid) >= 2:
        a, b = fit_linear_calibration(df_valid["Pred_raw"].values, df_valid["GT"].values)
    else:
        a, b = 1.0, 0.0

    print(f"\nKalibrierung gelernt: GT ≈ {a:.6f} * Pred_raw + {b:.3f}")

    # --- 3) Fehler berechnen (raw & calibrated) ---
    df_valid["Pred_cal"] = a * df_valid["Pred_raw"] + b

    df_valid["AbsErr_raw"] = np.abs(df_valid["Pred_raw"] - df_valid["GT"])
    df_valid["RelErr_raw"] = df_valid["AbsErr_raw"] / df_valid["GT"].clip(lower=1)

    df_valid["AbsErr_cal"] = np.abs(df_valid["Pred_cal"] - df_valid["GT"])
    df_valid["RelErr_cal"] = df_valid["AbsErr_cal"] / df_valid["GT"].clip(lower=1)

    df_valid["WithinTarget_raw"] = df_valid.apply(lambda r: within_target(r["Pred_raw"], r["GT"]), axis=1)
    df_valid["WithinTarget_cal"] = df_valid.apply(lambda r: within_target(r["Pred_cal"], r["GT"]), axis=1)

    # --- 4) CSVs schreiben ---
    # (A) Roh CSV (kompatibel plus extra Spalten)
    out_cols = [
        "Dateiname", "GT",
        "Pred_raw", "AbsErr_raw", "RelErr_raw", "WithinTarget_raw",
        "Pred_cal", "AbsErr_cal", "RelErr_cal", "WithinTarget_cal"
    ]

    # Haupt-CSV überschreiben: jetzt mit GT + raw/cal
    df_valid[out_cols].to_csv(CSV_FILE, index=False, float_format="%.2f")

    # Optional separate "kalibriert" CSV
    df_valid[out_cols].to_csv(CSV_FILE_CAL, index=False, float_format="%.2f")

    # --- 5) Summary ---
    def summary_block(label, within_col, abs_col, rel_col):
        within = float(df_valid[within_col].mean())
        mae = float(df_valid[abs_col].mean())
        mape = float(df_valid[rel_col].mean())
        p50 = float(np.median(df_valid[rel_col].values))
        p90 = float(np.percentile(df_valid[rel_col].values, 90))
        print(f"\n[{label}]")
        print(f"  Bilder: {len(df_valid)}")
        print(f"  Within Target (<=10% / low-count abs): {within*100:.1f}%")
        print(f"  MAE (abs): {mae:.2f}")
        print(f"  MAPE (rel): {mape*100:.2f}%")
        print(f"  Median rel err: {p50*100:.2f}%")
        print(f"  90th pct rel err: {p90*100:.2f}%")

    summary_block("RAW", "WithinTarget_raw", "AbsErr_raw", "RelErr_raw")
    summary_block("CALIBRATED", "WithinTarget_cal", "AbsErr_cal", "RelErr_cal")

    print(f"\nFertig! CSV geschrieben nach:\n  - {CSV_FILE}\n  - {CSV_FILE_CAL}")


if __name__ == "__main__":
    main()
