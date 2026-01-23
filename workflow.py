"""
Kompletter Workflow für das Bienenzähler-Projekt:
1. Trainingsdaten vorbereiten (inkl. Bilder resizen + Koordinaten anpassen)
2. Modell trainieren

WICHTIG: Die Bilder werden NICHT vorher resized! 
prepare_data.py lädt die Originalbilder, resized sie und passt die Koordinaten 
aus der annotations.xml automatisch an den gleichen Skalierungsfaktor an.
"""

import os
from prepare_data import prepare_training_data
from model import train_model


def step1_prepare_training_data(xml_path='test', 
                                images_folder='Wabenbilder',
                                output_folder='prepared_data'):
    """
    Step 1: Erstelle Trainingsdaten (X_train.npy und Y_train.npy).
    
    WICHTIG: Lädt ORIGINAL-Bilder, resized sie und passt die Koordinaten
    aus annotations.xml automatisch an. Die Koordinaten werden mit dem
    gleichen Skalierungsfaktor wie die Bilder transformiert!
    
    Args:
        xml_path: Pfad zu einer einzelnen .xml Datei ODER Ordner mit mehreren .xml Dateien
    """
    print("=" * 60)
    print("STEP 1: Trainingsdaten vorbereiten")
    print("=" * 60)
    print("(Bilder werden resized + Koordinaten angepasst)\n")
    
    if not os.path.exists(xml_path):
        print(f"Fehler: XML-Pfad nicht gefunden: {xml_path}")
        return False
    
    if not os.path.exists(images_folder):
        print(f"Fehler: Bildordner nicht gefunden: {images_folder}")
        return False
    
    try:
        X_train, Y_train = prepare_training_data(
            xml_path=xml_path,
            images_folder=images_folder,
            output_folder=output_folder,
            target_width=512,
            target_height=288,
            sigma=4
        )
        
        print(f"\n✓ Step 1 abgeschlossen: Daten vorbereitet")
        print(f"  X_train: {X_train.shape}")
        print(f"  Y_train: {Y_train.shape}\n")
        return True
        
    except Exception as e:
        print(f"Fehler bei der Datenvorbereitung: {e}")
        import traceback
        traceback.print_exc()
        return False


def step2_train_model(data_folder='prepared_data', epochs=50, batch_size=4):
    """
    Step 2: Trainiere das Modell.
    """
    print("=" * 60)
    print("STEP 2: Modell trainieren")
    print("=" * 60)
    
    if not os.path.exists(os.path.join(data_folder, 'X_train.npy')):
        print(f"Fehler: Trainingsdaten nicht gefunden in '{data_folder}'")
        return False
    
    try:
        model, history = train_model(
            data_folder=data_folder,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        
        if model:
            print(f"\n✓ Step 2 abgeschlossen: Modell trainiert")
            print(f"  Beste Modell: best_model.keras")
            print(f"  Finale Modell: final_model.keras\n")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_complete_workflow(skip_prepare=False, epochs=50, batch_size=4):
    """
    Führt den kompletten Workflow aus.
    
    WICHTIG: Verwendet die ORIGINAL-Bilder aus Wabenbilder/
    Die Bilder werden automatisch resized und die Koordinaten entsprechend angepasst!
    
    Args:
        skip_prepare: Überspringe Step 1 (wenn Daten bereits vorbereitet sind)
        epochs: Anzahl Trainingsepochen
        batch_size: Batch-Größe für Training
    """
    print("\n" + "=" * 60)
    print("BIENENZÄHLER - KOMPLETTER WORKFLOW")
    print("=" * 60 + "\n")
    
    # Step 1: Trainingsdaten vorbereiten (inkl. Resize + Koordinaten-Anpassung)
    if skip_prepare:
        print("Step 1 übersprungen (skip_prepare=True)\n")
    else:
        success = step1_prepare_training_data()
        if not success:
            print("❌ Workflow abgebrochen: Fehler in Step 1")
            return
    
    # Step 2: Modell trainieren
    success = step2_train_model(epochs=epochs, batch_size=batch_size)
    if not success:
        print("❌ Workflow abgebrochen: Fehler in Step 2")
        return
    
    # Fertig!
    print("=" * 60)
    print("✓ WORKFLOW ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 60)
    print("\nErstellt:")
    print("  - prepared_data/          (Trainingsdaten + resized Bilder)")
    print("  - best_model.keras        (bestes Modell)")
    print("  - final_model.keras       (finales Modell)")
    print("  - training_history.png    (Trainingsdiagramm)")
    print("  - predictions_sample.png  (Beispielvorhersagen)")
    print("\nNächste Schritte:")
    print("  1. Prüfe training_history.png für Trainingsverlauf")
    print("  2. Prüfe predictions_sample.png für Vorhersagequalität")
    print("  3. Verwende best_model.keras für Vorhersagen auf neuen Bildern")
    print("\nWICHTIG:")
    print("  Die Koordinaten aus annotations.xml wurden automatisch")
    print("  mit dem gleichen Faktor wie die Bilder skaliert!")


if __name__ == "__main__":
    # Konfiguration
    SKIP_PREPARE = False   # Auf True setzen, wenn Daten bereits vorbereitet sind
    EPOCHS = 100            # Anzahl Trainingsepochen
    BATCH_SIZE = 8         # Batch-Größe (bei wenig Daten klein halten)
    
    # Führe kompletten Workflow aus
    run_complete_workflow(
        skip_prepare=SKIP_PREPARE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
