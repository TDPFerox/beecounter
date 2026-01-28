# 🐝 BeeCounter - Dichtekarten-basierte Bienenzählung

Dieses Projekt nutzt ein **U-Net (Convolutional Neural Network)**, um Bienen auf hochauflösenden Wabenbildern zu lokalisieren und zu zählen. Statt einer Klassifizierung generiert das Modell eine **Dichtekarte (Density Map)**, die eine präzise räumliche Verteilung der Bienen ermöglicht.

## 🚀 Features
* **U-Net Architektur:** Symmetrischer Encoder-Decoder-Pfad für hochauflösende Merkmalsextraktion.
* **RAM-schonendes Streaming:** Ein spezialisierter `BeeDataGenerator` lädt Daten batchweise von der Festplatte, was das Training mit über 70.000 Kacheln auf Consumer-Hardware ermöglicht.
* **Kombinierter Loss:** Optimierung über Pixel-Dichte (MSE) und absolute Zählgenauigkeit (Count Loss).
* **Kontinuierliches Training:** Automatisches Speichern des besten Modells (`best_model.keras`) und Logging der Historie in einer CSV-Datei.

## 📂 Projektstruktur
* `workflow.py`: Zentrales Steuerungsskript für Datenvorbereitung und Training.
* `model.py`: Definition des U-Nets, der Loss-Funktionen und des Daten-Generators.
* `prepare_data.py`: Skript zur Kachelung (Tiling) und Augmentation der Rohbilder.
* `training_log.csv`: CSV-Protokoll aller Trainingsmetriken pro Epoche.

Zusätzlich existieren Skripte zur Generierung von Metriken und dem Predicten von unbekannten Testdaten.
Diese sind alle unter Src zu finden.

Im Bereich Modell sind die aktuellen mit meinen Trainingsdaten trainierten Modelle zu finden. Unter Metric finden sich Daten und Bilder von Trainingsläufen und Predictions.

## Aktueller Trainingslauf

![Trainingshistorie](/Metric/training_history.png)

Epoche 0 wurde hier abgeschnitten, da der Loss in der ersten Epoche so hoch war, das die nachfolgenden Epochen durch die Skalierung nicht mehr lesbar waren.

## 🛠 Installation & Setup

1.  **Umgebung einrichten:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install tensorflow pandas matplotlib opencv-python
    ```

2.  **Daten vorbereiten:**
    Um selber zu trainieren, müssen die Testdaten selbst zur Verfügung gestellt werden. Die Trainingsdaten müssen in einem Ordner Data abgelegt werden. Bilder kommen dabei in einen Ordner Data/Wabenbilder und die XML Dateien mit den Koordinaten für die Dichtekarten unter Data/annotations. Aktuell können nur Annotations basierend auf dem Format CVAT for images 1.1 eingelesen werden.

## 📈 Training ausführen

Starte den gesamten Prozess über die `workflow.py`:
```bash
python3 workflow.py