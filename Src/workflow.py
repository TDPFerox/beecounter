import os
from prepare_data import prepare_training_data
from model import train_model, plot_training_history

def step1_prepare_training_data():
    print("=" * 60)
    print("STEP 1: Trainingsdaten vorbereiten (STREAMING-MODUS)")
    print("=" * 60)

    base_images = 'Data/images'
    base_xml = 'Data/annotations'
    output = 'Data/prepared_data'
    
    if not os.path.exists(base_xml):
        print(f"Fehler: XML-Pfad nicht gefunden: {base_xml}")
        return False
    
    try:
        num_tiles = prepare_training_data(base_xml, base_images, output, mode='train')
        
        if num_tiles > 0:
            print(f"\n✓ Step 1 abgeschlossen: {num_tiles} Kacheln auf Festplatte gespeichert.\n")
        else:
            print("\n! Warnung: Es wurden keine gültigen Kacheln erzeugt.")
            return False
    except Exception as e:
        print(f"Fehler bei der Datenvorbereitung: {e}")
        return False
    
    try:
        num_tiles = prepare_training_data(base_xml, base_images, output, mode='val')
        
        if num_tiles > 0:
            print(f"\n✓ Step 1 abgeschlossen: {num_tiles} Kacheln auf Festplatte gespeichert.\n")
            return True
        else:
            print("\n! Warnung: Es wurden keine gültigen Kacheln erzeugt.")
            return False
    except Exception as e:
        print(f"Fehler bei der Datenvorbereitung: {e}")
        return False
    

def step2_train_model(continue_training, data_folder='Data/prepared_data', epochs=100, batch_size=4):
    print("=" * 60)
    print("STEP 2: Modell trainieren (Data Generator)")
    print("=" * 60)
    
    try:
        history = train_model(
            continue_training,
            data_folder=data_folder,
            epochs=epochs,
            batch_size=batch_size
        )
        
        if history:
            print("\nErstelle Trainings-Diagramme...")
            plot_training_history(history)
            return True
        return False
    except Exception as e:
        print(f"Fehler beim Training: {e}")
        return False

def run_complete_workflow(skip_prepare=False, continue_training=False, epochs=100, batch_size=16):
    print("\n" + "=" * 60)
    print("BIENENZÄHLER - RAM-SCHONENDER WORKFLOW")
    print("=" * 60 + "\n")
    
    # 1. Daten vorbereiten (optional)
    if not skip_prepare:
        if not step1_prepare_training_data(): return
    
    # 2. Training starten
    if not step2_train_model(continue_training, epochs=epochs, batch_size=batch_size): return
    
    print("\n✓ WORKFLOW ERFOLGREICH ABGESCHLOSSEN!")

if __name__ == "__main__":
    run_complete_workflow(skip_prepare=False, continue_training=True, epochs=100, batch_size=32)