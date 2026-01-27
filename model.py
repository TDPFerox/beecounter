import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.utils import Sequence
import glob

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

def combined_loss(y_true, y_pred, lambda_count=0.5):
    # 1. Density Loss (Punktgenauigkeit)
    density_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # 2. Count Loss (QUADRIERT für mehr Druck bei hohen Zahlen)
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    
    # MSE statt MAE beim Zähler
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
    def __init__(self, tile_indices, tiles_dir, batch_size=4):
        self.tile_indices = tile_indices
        self.tiles_dir = tiles_dir
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.tile_indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.tile_indices[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = [], []
        
        for idx in indices:
            X.append(np.load(os.path.join(self.tiles_dir, f"x_{idx}.npy")))
            y_data = np.load(os.path.join(self.tiles_dir, f"y_{idx}.npy"))
            Y.append(np.expand_dims(y_data, axis=-1))
            
        return np.array(X), np.array(Y)


def train_model(continue_training, data_folder='prepared_data', epochs=50, batch_size=4):
    tiles_dir = os.path.join(data_folder, 'tiles')
    
    # 1. Alle verfügbaren Kacheln finden
    x_files = glob.glob(os.path.join(tiles_dir, 'x_*.npy'))
    num_tiles = len(x_files)
    
    if num_tiles == 0:
        print("Fehler: Keine Kacheln in 'prepared_data/tiles' gefunden!")
        return None, None

    # 2. Indizes mischen und aufteilen (80% Training, 20% Validierung)
    indices = np.arange(num_tiles)
    np.random.shuffle(indices)
    split = int(0.8 * num_tiles)
    train_idx, val_idx = indices[:split], indices[split:]
    
    print(f"Starte Training mit {len(train_idx)} Kacheln und {len(val_idx)} zur Validierung.")

    # 3. Generatoren erstellen
    train_gen = BeeDataGenerator(train_idx, tiles_dir, batch_size)
    val_gen = BeeDataGenerator(val_idx, tiles_dir, batch_size)
    
    # In model.py innerhalb von train_model:
    if continue_training:
        print("Lade existierendes Modell für weiteres Training...")
        model = tf.keras.models.load_model('best_model.keras', custom_objects={'combined_loss': combined_loss, 'count_loss': count_loss})
    else:
        print("Baue neues Modell...")
        model = build_bee_counter()
    # 4. Modell bauen und kompilieren
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=["mae", count_loss]
        )
        
    # 5. Callbacks (wie gehabt)
    early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    check = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    # 6. Training mit Generator starten
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early, check],
        verbose=1,
        workers=4,                # Nutzt 4 CPU-Kerne zum Laden der Daten
        use_multiprocessing=True, # Erlaubt paralleles Laden
        max_queue_size=10
    )
    
    model.save('final_model.keras')
    return model, history


def plot_training_history(history):
    """Visualisiert den Trainingsverlauf."""
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
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
    plt.savefig('training_history.png', dpi=150)
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
        data_folder='prepared_data',
        epochs=50,
        batch_size=4,
        test_split=0.15,      # 15% für Test
        validation_split=0.15  # 15% für Validation (=> 70% Training)
    )
    
    if model:
        print("\nTraining abgeschlossen!")
        print("Verwende das Modell mit: model = tf.keras.models.load_model('best_model.keras')")