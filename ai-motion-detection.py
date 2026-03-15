import gi
import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Silenzia i log di TensorFlow e avvisi vari
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

Gst.init(None)

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    "video": {
        "width": 640,
        "height": 360,
        "fps": 20,
        "udp_port": 5000,
        "output_dir": "videos"
    },
    "motion": {
        "threshold": 250,
        "min_area": 1200,
        "cooldown": 2.0,
        "history": 500
    },
    "ai": {
        "model_path": "yolo26n.pt",
        "conf_threshold": 0.4,
        "recognition_delay": 1.0,
        "crop_padding": 40
    }
}

class VideoRecorder:
    def __init__(self, config):
        self.output_dir = config["output_dir"]
        self.fps = config["fps"]
        self.res = (config["width"], config["height"])
        self.writer = None
        self.temp_path = ""
        self.start_ts = ""
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start(self):
        self.start_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.temp_path = os.path.join(self.output_dir, f"temp_{self.start_ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.temp_path, fourcc, self.fps, self.res)
        return self.start_ts

    def write(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop(self, detections):
        if self.writer:
            self.writer.release()
            self.writer = None
            obj_name = "-".join(detections[:3]) if detections else "motion"
            final_path = os.path.join(self.output_dir, f"{self.start_ts}_{obj_name}.mp4")
            if os.path.exists(self.temp_path):
                os.rename(self.temp_path, final_path)
                print(f"[v] Saved: {final_path}")

class VisionEngine:
    def __init__(self, config):
        self.cfg = config
        print(f"[*] AI Engine active: {self.cfg['model_path']}")
        self.model = YOLO(self.cfg['model_path'])

    def detect(self, frame, contours):
        if not contours: return None
        
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        pad = self.cfg["crop_padding"]
        
        h_max, w_max = frame.shape[:2]
        y1, y2 = max(0, y-pad), min(h_max, y+h+pad)
        x1, x2 = max(0, x-pad), min(w_max, x+w+pad)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size > 0:
            results = self.model(crop, verbose=False, conf=self.cfg["conf_threshold"])
            for r in results:
                if len(r.boxes) > 0:
                    return self.model.names[int(r.boxes[0].cls[0])]
        return None

class MotionMonitorApp:
    def __init__(self, config):
        self.cfg = config
        self.vision = VisionEngine(self.cfg["ai"])
        self.recorder = VideoRecorder(self.cfg["video"])
        
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg["motion"]["history"], 
            varThreshold=self.cfg["motion"]["threshold"]
        )
        
        self.current_frame = None
        self.is_recording = False
        self.yolo_done = False
        self.detections = []
        self.first_motion_time = 0
        self.last_motion_time = 0
        self.last_save_time = 0

        self._init_gst()

    def _init_gst(self):
        v = self.cfg["video"]
        pipe_str = (
            f"udpsrc port={v['udp_port']} caps=\"application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96\" "
            f"! rtph264depay ! h264parse ! avdec_h264 ! videoconvert "
            f"! videoscale ! video/x-raw,width={v['width']},height={v['height']} ! videoconvert ! video/x-raw,format=BGR "
            f"! appsink name=sink emit-signals=True sync=false"
        )
        self.pipeline = Gst.parse_launch(pipe_str)
        self.pipeline.get_by_name("sink").connect("new-sample", self._on_sample)

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w, h = caps.get_structure(0).get_value("width"), caps.get_structure(0).get_value("height")
        res, map_info = buf.map(Gst.MapFlags.READ)
        if res:
            self.current_frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy()
            buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        m_cfg = self.cfg["motion"]
        ai_cfg = self.cfg["ai"]
        
        try:
            while True:
                if self.current_frame is None: continue
                
                now = time.time()
                # Creiamo il frame base e aggiungiamo IMMEDIATAMENTE il timestamp
                frame = self.current_frame.copy()
                
                # --- AGGIUNTA DATA E ORA ---
                timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                # Posizionato in basso a sinistra (x=10, y=altezza-10)
                cv2.putText(frame, timestamp_str, (10, self.cfg["video"]["height"] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 1. Motion Logic
                self.back_sub.setVarThreshold(m_cfg["threshold"])
                mask = self.back_sub.apply(frame)
                _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in contours if cv2.contourArea(c) > m_cfg["min_area"]]

                # 2. State Machine
                if valid_cnts:
                    if not self.is_recording:
                        self.is_recording = True
                        self.first_motion_time = now
                        self.recorder.start()
                    self.last_motion_time = now

                if self.is_recording:
                    if not self.yolo_done and (now - self.first_motion_time >= ai_cfg["recognition_delay"]):
                        label = self.vision.detect(frame, valid_cnts)
                        if label:
                            self.detections.append(label)
                            self.yolo_done = True

                    if (now - self.last_save_time) >= (1.0 / self.recorder.fps):
                        self.recorder.write(frame)
                        self.last_save_time = now
                    
                    if (now - self.last_motion_time) > m_cfg["cooldown"]:
                        self.recorder.stop(self.detections)
                        self.is_recording = False
                        self.yolo_done = False
                        self.detections = []

                # Render UI (passiamo il frame già marchiato col timestamp)
                self._show_ui(frame, valid_cnts, now)

                # 3. Key Handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 82: # SU
                    m_cfg["threshold"] += 50
                elif key == 84: # GIU
                    m_cfg["threshold"] = max(10, m_cfg["threshold"] - 50)
                elif key == 83: # DX
                    m_cfg["min_area"] += 100
                elif key == 81: # SX
                    m_cfg["min_area"] = max(100, m_cfg["min_area"] - 100)

        finally:
            self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()

    def _show_ui(self, frame, contours, now):
        ui = frame.copy()
        cv2.rectangle(ui, (0, 0), (300, 85), (0, 0, 0), -1)
        if contours: cv2.drawContours(ui, contours, -1, (0, 255, 0), -1)
        cv2.addWeighted(ui, 0.3, frame, 0.7, 0, frame)
        
        status = self.detections[0].upper() if self.detections else "SCANNING..."
        if self.is_recording and not self.yolo_done:
            status = f"WAITING... ({int(now-self.first_motion_time)}s)"
            
        m_cfg = self.cfg["motion"]
        cv2.putText(frame, f"THR: {m_cfg['threshold']} | AREA: {m_cfg['min_area']}", (10, 30), 1, 1, (255, 255, 0), 1)
        cv2.putText(frame, f"DETECTED: {status}", (10, 65), 1, 1, (0, 255, 255), 1)
        
        if self.is_recording: 
            cv2.circle(frame, (610, 30), 7, (0, 0, 255), -1)
            
        cv2.imshow("AI Monitor", frame)

if __name__ == "__main__":
    app = MotionMonitorApp(CONFIG)
    app.run()
