import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, CSVLogger)
from tensorflow.keras.utils import Sequence
import glob
import pandas as pd

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def count_loss(y_true, y_pred):
    """
    Berechnet den absoluten Fehler zwischen der Summe der Ground Truth
    und der Summe der Vorhersage (= Anzahl der gezählten Bienen).
    """
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean(tf.abs(true_count - pred_count))

def combined_loss(y_true, y_pred, lambda_count=50.0):
    # 1. Density Loss (Punktgenauigkeit)
    density_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    
    count_loss_val = tf.reduce_mean(tf.square(true_count - pred_count))

    return density_loss + (lambda_count * count_loss_val)


def build_bee_counter(input_shape=(None, None, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    b = conv_block(p3, 256)

    # Decoder
    u3 = UpSampling2D()(b)
    u3 = Concatenate()([u3, c3])
    c4 = conv_block(u3, 128)

    u2 = UpSampling2D()(c4)
    u2 = Concatenate()([u2, c2])
    c5 = conv_block(u2, 64)

    u1 = UpSampling2D()(c5)
    u1 = Concatenate()([u1, c1])
    c6 = conv_block(u1, 32)

    # Output: Dichtekarte
    output = Conv2D(1, 1, activation="relu")(c6)

    return Model(inputs, output)

class BeeDataGenerator(Sequence):
    def __init__(self, file_list, batch_size=4, shuffle=False):
        self.file_list = file_list # Liste der vollen Pfade zu den x_*.npy Dateien
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def on_epoch_end(self):
        # Wichtig für das Training: Nach jeder Epoche neu mischen
        if self.shuffle:
            np.random.shuffle(self.file_list)

    def __getitem__(self, index):
        batch_files = self.file_list[index*self.batch_size : (index+1)*self.batch_size]
        X, Y = [], []
        
        for x_path in batch_files:
            # Lade X (Bildkachel)
            X.append(np.load(x_path))
            
            # Pfad für Y (Dichtekarte) ableiten: x_0.npy -> y_0.npy im selben Ordner
            y_path = x_path.replace('x_', 'y_')
            y_data = np.load(y_path)
            Y.append(np.expand_dims(y_data, axis=-1))
            
        return np.array(X), np.array(Y)


def train_model(continue_training, data_folder='Data/prepared_data', epochs=50, batch_size=4):
    # Pfade zu den neuen Unterordnern
    train_dir = os.path.join(data_folder, 'train')
    val_dir = os.path.join(data_folder, 'val')
    history_file = 'Metric/training_log.csv'
    
    # 1. Daten direkt aus den Ordnern laden
    train_files = glob.glob(os.path.join(train_dir, 'x_*.npy'))
    val_files = glob.glob(os.path.join(val_dir, 'x_*.npy'))
    
    if not train_files:
        print(f"Fehler: Keine Trainingsdaten in {train_dir} gefunden!")
        return None, None

    # Generatoren initialisieren
    # Training: shuffle=True | Validierung: shuffle=False
    train_gen = BeeDataGenerator(train_files, batch_size, shuffle=True)
    val_gen = BeeDataGenerator(val_files, batch_size, shuffle=False)
    
    print(f"Training mit {len(train_files)} Kacheln")
    print(f"Validierung mit {len(val_files)} Kacheln")

    # 2. Modell laden oder bauen (bleibt fast gleich)
    if continue_training and os.path.exists('Model/best_model.keras'):
        print("Lade existierendes Modell...")
        model = tf.keras.models.load_model('Model/best_model.keras', 
                custom_objects={'combined_loss': combined_loss, 'count_loss': count_loss},
                compile=False)
        
        # Mit niedriger Lernrate für Fine-Tuning neu kompilieren
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                    loss=combined_loss, metrics=["mae", count_loss])
    else:
        print("Baue neues Modell...")
        model = build_bee_counter()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=combined_loss, metrics=["mae", count_loss])
        
    # 3. Callbacks (bleiben gleich)
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    check = ModelCheckpoint('Model/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    log_csv = CSVLogger(history_file, separator=',', append=continue_training)
    
    # 4. Training starten
    initial_epoch = 0
    if continue_training and os.path.exists(history_file):
        try:
            existing_history = pd.read_csv(history_file)
            if not existing_history.empty:
                initial_epoch = int(existing_history['epoch'].max()) + 1
                print(f"Setze Training fort bei Epoche {initial_epoch}...")
        except Exception as e:
            print(f"Konnte History nicht lesen: {e}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[early, check, log_csv, reduce_lr],
        verbose=1
    )
    
    # 5. Rückgabe (bleibt gleich)
    if os.path.exists(history_file):
        full_df = pd.read_csv(history_file)
        class HistoryWrapper:
            def __init__(self, data_dict): self.history = data_dict
        return model, HistoryWrapper(full_df.to_dict(orient='list'))
    
    return model, history


def plot_training_history(history):
    
    start_epoch = 1 # Überspringe die allererste Epoche
    epochs_range = range(start_epoch, len(history.history['loss']))
    
    """Visualisiert den Trainingsverlauf."""
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    #plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training und Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Mean Absolute Error')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['count_loss'], label='Training Count Loss')
    plt.plot(history.history['val_count_loss'], label='Validation Count Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Count Loss (Anzahl Bienen)')
    plt.legend()
    plt.title('Count Loss (Zählfehler)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Metric/training_history.png', dpi=150)
    plt.close()
    print("Trainingsverlauf gespeichert als 'training_history.png'")


def visualize_predictions(model, X_samples, Y_true, prefix=''):
    """Visualisiert Vorhersagen auf Beispielbildern."""
    Y_pred = model.predict(X_samples, verbose=0)
    
    n_samples = len(X_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Originalbild
        axes[i, 0].imshow(X_samples[i])
        axes[i, 0].set_title('Original Bild')
        axes[i, 0].axis('off')
        
        # Ground Truth Dichtekarte
        true_count = np.sum(Y_true[i])
        axes[i, 1].imshow(Y_true[i, :, :, 0], cmap='hot', interpolation='bilinear')
        axes[i, 1].set_title(f'Ground Truth\n(~{int(true_count)} Bienen)')
        axes[i, 1].axis('off')
        
        # Vorhersage Dichtekarte
        pred_count = np.sum(Y_pred[i])
        error = abs(pred_count - true_count)
        axes[i, 2].imshow(Y_pred[i, :, :, 0], cmap='hot', interpolation='bilinear')
        axes[i, 2].set_title(f'Vorhersage\n(~{int(pred_count)} Bienen, Fehler: {int(error)})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    filename = f'predictions_{prefix}_sample.png' if prefix else 'predictions_sample.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Beispielvorhersagen gespeichert als '{filename}'")


if __name__ == "__main__":
    # Trainiere das Modell
    model, history = train_model(
        data_folder='Data/prepared_data',
        epochs=50,
        batch_size=4,
        test_split=0.15,      # 15% für Test
        validation_split=0.15  # 15% für Validation (=> 70% Training)
    )
    
    if model:
        print("\nTraining abgeschlossen!")
        print("Verwende das Modell mit: model = tf.keras.models.load_model('best_model.keras')")