import os
from prepare_data import prepare_training_data
from model import train_model

def step1_prepare_training_data(xml_path='test', 
                                images_folder='Wabenbilder',
                                output_folder='prepared_data'):
    print("=" * 60)
    print("STEP 1: Trainingsdaten vorbereiten (TILING & AUGMENTATION)")
    print("=" * 60)
    
    if not os.path.exists(xml_path):
        print(f"Fehler: XML-Pfad nicht gefunden: {xml_path}")
        return False
    
    try:
        # Hier wurden die veralteten Argumente target_width/height entfernt
        X_train, Y_train = prepare_training_data(
            xml_path=xml_path,
            images_folder=images_folder,
            output_folder=output_folder,
            tile_size=256 # Die neue Kachelgröße
        )
        
        print(f"\n✓ Step 1 abgeschlossen: {len(X_train)} Kacheln vorbereitet\n")
        return True
    except Exception as e:
        print(f"Fehler bei der Datenvorbereitung: {e}")
        return False

def step2_train_model(data_folder='prepared_data', epochs=100, batch_size=16):
    print("=" * 60)
    print("STEP 2: Modell trainieren")
    print("=" * 60)
    
    try:
        # Training mit den neuen Kacheln
        model, history = train_model(
            data_folder=data_folder,
            epochs=epochs,
            batch_size=batch_size
        )
        return True if model else False
    except Exception as e:
        print(f"Fehler beim Training: {e}")
        return False

def run_complete_workflow(skip_prepare=False, epochs=100, batch_size=16):
    print("\n" + "=" * 60)
    print("BIENENZÄHLER - TILING WORKFLOW")
    print("=" * 60 + "\n")
    
    if not skip_prepare:
        if not step1_prepare_training_data(): return
    
    if not step2_train_model(epochs=epochs, batch_size=batch_size): return
    
    print("\n✓ WORKFLOW ERFOLGREICH ABGESCHLOSSEN!")

if __name__ == "__main__":
    run_complete_workflow(skip_prepare=False, epochs=100, batch_size=8)