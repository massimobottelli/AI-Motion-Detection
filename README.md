# 🛡️ AI Motion Guard: Smart Detection & Recognition

**AI Motion Guard** è un sistema di videosorveglianza intelligente che combina la velocità della **Motion Detection** tradizionale con la precisione del Deep Learning tramite **YOLO**. 

A differenza dei sistemi standard, questa applicazione riduce drasticamente i falsi positivi analizzando il contenuto video solo quando viene rilevato un movimento reale e attendendo il momento ottimale (1 secondo di delay) per identificare l'oggetto, garantendo che sia ben visibile nell'inquadratura.

---

## ✨ Caratteristiche Principali

* **Rilevamento Ibrido:** Motion detection ultra-veloce (OpenCV MOG2) unita a Object Recognition (YOLO).
* **Analisi Ritardata (Smart Trigger):** L'IA interviene dopo 1 secondo dall'inizio del movimento per catturare l'immagine migliore.
* **Registrazione Automatica:** Salva file `.mp4` solo durante l'evento, ottimizzando lo spazio su disco.
* **Auto-Naming:** I video vengono rinominati automaticamente con il timestamp e l'oggetto riconosciuto (es. `20260307-1830_person.mp4`).
* **Architettura Modulare:** Codice organizzato in classi separate per Visione, Registrazione e Monitoraggio.
* **Streaming UDP:** Supporto nativo per flussi video remoti (es. da un Raspberry Pi).

---

## 🛠️ Requisiti

### Software
* **Python 3.9+**
* **GStreamer 1.0** (con plugin `ugly` e `good` per il supporto H.264)
* Librerie Python:
    ```bash
    pip install opencv-python numpy ultralytics PyGObject
    ```

### Hardware (Consigliato)
* **IP Camera** o **Raspberry Pi** con webcam USB.
* PC con CPU moderna o GPU NVIDIA (opzionale) per un'inferenza YOLO più veloce.

---

## 🚀 Installazione e Avvio

1. **Clona il repository:**
   ```bash
   git clone [https://github.com/tuo-username/ai-motion-guard.git](https://github.com/tuo-username/ai-motion-guard.git)
   cd ai-motion-guard
Configura il flusso video:
Assicurati che la tua sorgente video trasmetta sulla porta UDP 5000.

Avvia l'applicazione:

Bash
python main.py
⚙️ Configurazione (Parametri)
I parametri principali sono centralizzati nel dizionario CONFIG all'interno del codice:

Parametro	Descrizione	Default
threshold	Sensibilità al movimento (più basso = più sensibile)	250
min_area	Dimensione minima in pixel dell'oggetto per attivare la REC	1500
recognition_delay	Secondi di attesa prima dell'intervento di YOLO	1.0
cooldown	Secondi di assenza di movimento prima di chiudere il video	2.0
🎥 Streaming da Raspberry Pi
Per inviare il video dal tuo Raspberry Pi al sistema, usa questo comando:

Bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    video/x-raw,width=640,height=360,framerate=20/1 ! \
    videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! \
    rtph264pay config-interval=1 ! \
    udpsink host=INDIRIZZO_IP_PC port=5000
⌨️ Comandi Rapidi (Hotkeys)
Durante l'esecuzione, puoi interagire con la dashboard:

Q: Esci dall'applicazione.

S: Salva i parametri attuali.

Freccia SU/GIÙ: Regola la soglia di sensibilità.

Freccia DX/SX: Regola l'area minima di rilevamento.
