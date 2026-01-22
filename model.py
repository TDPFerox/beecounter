import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)

def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_bee_counter(input_shape=(288, 512, 3)):
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
    output = Conv2D(1, 1, activation="linear")(c6)

    return Model(inputs, output)


def train_model(data_folder='prepared_data', epochs=50, batch_size=4, validation_split=0.2):
    """
    Trainiert das Bienenzähler-Modell mit den vorbereiteten Daten.
    
    Args:
        data_folder: Ordner mit X_train.npy und Y_train.npy
        epochs: Anzahl der Trainingsepochen
        batch_size: Batch-Größe für Training
        validation_split: Anteil der Daten für Validierung
    """
    # Lade Trainingsdaten
    print("Lade Trainingsdaten...")
    X_train_path = os.path.join(data_folder, 'X_train.npy')
    Y_train_path = os.path.join(data_folder, 'Y_train.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(Y_train_path):
        print("Fehler: Trainingsdaten nicht gefunden!")
        print("Bitte führe zuerst 'prepare_data.py' aus.")
        return None
    
    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)
    
    print(f"X_train Shape: {X_train.shape}")
    print(f"Y_train Shape: {Y_train.shape}")
    print(f"Anzahl Bilder: {len(X_train)}")
    
    # Erstelle Modell
    print("\nErstelle Modell...")
    model = build_bee_counter()
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    
    # Callbacks
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    check = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    # Training
    print(f"\nStarte Training für {epochs} Epochen...")
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early, check],
        verbose=1
    )
    
    # Speichere finales Modell
    model.save('final_model.keras')
    print("\nModell gespeichert als 'final_model.keras' und 'best_model.keras'")
    
    # Visualisiere Training
    plot_training_history(history)
    
    # Teste auf einigen Beispielen
    visualize_predictions(model, X_train[:3], Y_train[:3])
    
    return model, history


def plot_training_history(history):
    """Visualisiert den Trainingsverlauf."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training und Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Mean Absolute Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("Trainingsverlauf gespeichert als 'training_history.png'")


def visualize_predictions(model, X_samples, Y_true):
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
        axes[i, 2].imshow(Y_pred[i, :, :, 0], cmap='hot', interpolation='bilinear')
        axes[i, 2].set_title(f'Vorhersage\n(~{int(pred_count)} Bienen)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_sample.png', dpi=150)
    plt.close()
    print("Beispielvorhersagen gespeichert als 'predictions_sample.png'")


if __name__ == "__main__":
    # Trainiere das Modell
    model, history = train_model(
        data_folder='prepared_data',
        epochs=50,
        batch_size=8,
        validation_split=0.2
    )
    
    if model:
        print("\nTraining abgeschlossen!")
        print("Verwende das Modell mit: model = tf.keras.models.load_model('best_model.keras')")