import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_from_csv(file_path='Metric/training_log.csv'):
    if not os.path.exists(file_path):
        print(f"Fehler: {file_path} nicht gefunden.")
        return

    df = pd.read_csv(file_path)
    
    df_plot = df.iloc[1:].copy()
    
    epochs = df_plot['epoch']

    plt.figure(figsize=(18, 5))

    # --- Plot 1: Loss ---
    plt.subplot(1, 3, 1)
    plt.plot(epochs, df_plot['loss'], label='Training Loss', marker='o', markersize=3)
    plt.plot(epochs, df_plot['val_loss'], label='Validation Loss', marker='o', markersize=3)
    plt.title('Loss (ab Epoche 1)')
    plt.xlabel('Epoche')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: MAE ---
    plt.subplot(1, 3, 2)
    plt.plot(epochs, df_plot['mae'], label='Training MAE', marker='o', markersize=3)
    plt.plot(epochs, df_plot['val_mae'], label='Validation MAE', marker='o', markersize=3)
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoche')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 3: Count Loss (Der Zählfehler) ---
    plt.subplot(1, 3, 3)
    plt.plot(epochs, df_plot['count_loss'], label='Training Count Loss', marker='o', markersize=3)
    plt.plot(epochs, df_plot['val_count_loss'], label='Validation Count Loss', marker='o', markersize=3)
    plt.title('Zählfehler (Anzahl Bienen)')
    plt.xlabel('Epoche')
    plt.ylabel('Abweichung in Stück')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Metric/training_history.png')
    print("Diagramm 'training_history.png' wurde erstellt (Epoche 0 übersprungen).")
    plt.show()

if __name__ == "__main__":
    plot_from_csv()