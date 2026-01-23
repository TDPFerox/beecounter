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

def combined_loss(y_true, y_pred, lambda_count=0.1):
    # Density Loss
    density_loss = tf.reduce_mean(tf.abs(y_true - y_pred))  # L1 statt MSE

    # Count Loss
    true_count = tf.reduce_sum(y_true, axis=[1, 2, 3])
    pred_count = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    count_loss = tf.reduce_mean(tf.abs(true_count - pred_count))

    return density_loss + lambda_count * count_loss


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


def train_model(data_folder='prepared_data', epochs=50, batch_size=16, test_split=0.15, validation_split=0.15):
    """
    Trainiert das Bienenzähler-Modell mit den vorbereiteten Daten.
    
    Args:
        data_folder: Ordner mit X_train.npy und Y_train.npy
        epochs: Anzahl der Trainingsepochen
        batch_size: Batch-Größe für Training
        test_split: Anteil der Daten für Test-Set (Standard: 15%)
        validation_split: Anteil der Daten für Validierung (Standard: 15%)
    """
    # In model.py ganz oben
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Lade Trainingsdaten
    print("Lade Trainingsdaten...")
    X_train_path = os.path.join(data_folder, 'X_train.npy')
    Y_train_path = os.path.join(data_folder, 'Y_train.npy')
    
    if not os.path.exists(X_train_path) or not os.path.exists(Y_train_path):
        print("Fehler: Trainingsdaten nicht gefunden!")
        print("Bitte führe zuerst 'prepare_data.py' aus.")
        return None
    
    X_data = np.load(X_train_path)
    Y_data = np.load(Y_train_path)
    
    print(f"Gesamtdaten Shape: X={X_data.shape}, Y={Y_data.shape}")
    print(f"Anzahl Bilder: {len(X_data)}")
    
    # Daten aufteilen: Train / Validation / Test
    from sklearn.model_selection import train_test_split
    
    # Zuerst Test-Set abtrennen
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_data, Y_data, 
        test_size=test_split, 
        random_state=42
    )
    
    # Dann Validation-Set von verbleibenden Daten
    val_size_adjusted = validation_split / (1 - test_split)  # Anteil vom verbleibenden Set
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val,
        test_size=val_size_adjusted,
        random_state=42
    )
    
    print(f"\n📊 Datenaufteilung:")
    print(f"   Training:   {len(X_train)} Bilder ({len(X_train)/len(X_data)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} Bilder ({len(X_val)/len(X_data)*100:.1f}%)")
    print(f"   Test:       {len(X_test)} Bilder ({len(X_test)/len(X_data)*100:.1f}%)")
    
    # Erstelle Modell
    print("\nErstelle Modell...")
    model = build_bee_counter()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[
            "mae",
            count_loss
        ]
    )
    model.summary()
    
    # Callbacks
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    check = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    # Training mit explizitem Validation-Set
    print(f"\nStarte Training für {epochs} Epochen...")
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),  # Explizites Validation-Set
        callbacks=[early, check],
        verbose=2
    )
    
    # Finale Evaluation auf Test-Set
    print("\n" + "="*60)
    print("📈 FINALE EVALUATION AUF TEST-SET")
    print("="*60)
    test_loss, test_mae, test_count_loss = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss (MSE):     {test_loss:.4f}")
    print(f"Test MAE:            {test_mae:.4f}")
    print(f"Test Count Loss:     {test_count_loss:.2f} Bienen")
    print("="*60)
    
    # Speichere finales Modell
    model.save('final_model.keras')
    print("\nModell gespeichert als 'final_model.keras' und 'best_model.keras'")
    
    # Visualisiere Training
    plot_training_history(history)
    
    # Teste auf Beispielen aus Test-Set
    visualize_predictions(model, X_test[:3], Y_test[:3], prefix='test')
    
    # Teste auch auf Training-Beispielen zum Vergleich
    visualize_predictions(model, X_train[:3], Y_train[:3], prefix='train')
    
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