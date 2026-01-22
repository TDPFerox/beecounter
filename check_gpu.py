"""
Prüft ob TensorFlow die GPU erkennt und nutzen kann.
"""

import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("\n" + "=" * 60)
print("GPU-Verfügbarkeit:")
print("=" * 60)

# Prüfe ob GPU verfügbar ist
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ {len(gpus)} GPU(s) gefunden:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        # Zeige GPU-Details
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details}")
        except:
            pass
    
    # GPU Memory Growth aktivieren (verhindert dass TF allen Speicher belegt)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n✓ GPU Memory Growth aktiviert")
    except RuntimeError as e:
        print(f"\nWarnung: {e}")
else:
    print("❌ Keine GPU gefunden - TensorFlow läuft auf CPU")
    print("\nUm GPU zu nutzen:")
    print("  1. Installiere CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  2. Installiere cuDNN: https://developer.nvidia.com/cudnn")
    print("  3. Installiere tensorflow mit GPU-Support:")
    print("     pip install tensorflow[and-cuda]")

print("\n" + "=" * 60)
print("Verfügbare Geräte:")
print("=" * 60)
for device in tf.config.list_logical_devices():
    print(f"  {device.device_type}: {device.name}")

print("\n" + "=" * 60)
print("CUDA-Build-Informationen:")
print("=" * 60)
print(f"CUDA verfügbar: {tf.test.is_built_with_cuda()}")
print(f"GPU verfügbar: {tf.test.is_gpu_available(cuda_only=False) if hasattr(tf.test, 'is_gpu_available') else 'N/A (deprecated)'}")

# Kleiner GPU-Test
if gpus:
    print("\n" + "=" * 60)
    print("GPU-Test:")
    print("=" * 60)
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("Matrix-Multiplikation auf GPU erfolgreich:")
        print(c.numpy())
