import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import math
from pypylon import pylon
import pytesseract
from pylibdmtx.pylibdmtx import decode
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Configure Tesseract path (update to your installation path)
pytesseract.pytesseract.tesseract_cmd = r''

class VisionCameraApp:
    def __init__(self, root):
        # Frame queue and thread pool for async tasks
        self.frame_queue = queue.Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        self.root = root
        self.root.title("Basler Camera Vision Demo")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Display throttling
        self.display_last = time.time()
        self.display_fps_interval = 1/30

        # White balance gain and calibration flag
        self.wb_gain = np.ones(3, dtype=np.float32)
        self.calibrating = False

        # Processing parameters
        self.params = {
            'edge': (50, 150),
            'thresh': 128,
            'pattern_thresh': 0.7,
            'blob_thresh': 128
        }
        # Blob-specific tunable params
        self.blob_params = {
            'minArea': 50,
            'maxArea': 5000,
            'minCircularity': 0.5,
            'minInertiaRatio': 0.5
        }
        self.template = None
        self.tpl_size = (0, 0)

        # Recording
        self.recording = False
        self.writer = None

        # Object tracking
        self.tracker = None
        self.bbox = None
        self.is_selecting_roi = False
        self.start_x = self.start_y = -1

        # Initial blob detector
        self.blob_detector = self.make_blob_detector()

        # Processing modes
        self.processing_mode = tk.StringVar(value="None")
        self.modes = [
            ("None", "None"),
            ("Edge Detection", "EdgeDetection"),
            ("Thresholding", "Thresholding"),
            ("Pattern Matching", "PatternMatching"),
            ("Object Tracking", "ObjectTracking"),
            ("Line Detection", "LineDetection"),
            ("Blurring", "Blurring"),
            ("Histogram Equalization", "HistogramEqualization"),
            ("Contour Detection", "ContourDetection"),
            ("Blob Detection", "BlobDetection")
        ]

        # Build UI
        self.build_ui()
        # Connect camera
        self.connect_camera()
        if not self.is_camera_open:
            messagebox.showerror("Error", "Cannot open Basler camera. Exiting.")
            root.destroy()
            return

        # Camera resolution setup
        self.current_width = self.camera.Width.GetValue()
        self.current_height = self.camera.Height.GetValue()
        self.setup_resolutions()

        # Bind events
        self.video_panel.bind("<ButtonPress-1>", self.on_mouse_down)
        self.video_panel.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_panel.bind("<ButtonRelease-1>", self.on_mouse_up)

        # FPS tracking
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Start grab thread and display loop
        threading.Thread(target=self._grab_thread, daemon=True).start()
        self.grab_loop()

    def build_ui(self):
        # Video panel
        self.video_panel = tk.Label(self.root, bg='black')
        self.video_panel.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Resolution controls
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="Resolution:").pack(side=tk.LEFT)
        self.res_var = tk.StringVar()
        self.res_combo = ttk.Combobox(ctrl, textvariable=self.res_var, state='readonly', width=12)
        self.res_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Apply", command=self.apply_resolution).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(ctrl, text="W:").pack(side=tk.LEFT)
        self.w_entry = ttk.Entry(ctrl, width=6); self.w_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="H:").pack(side=tk.LEFT)
        self.h_entry = ttk.Entry(ctrl, width=6); self.h_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Set Custom", command=self.apply_custom_resolution).pack(side=tk.LEFT)

        # Action buttons
        action = ttk.Frame(self.root)
        action.pack(fill=tk.X, padx=10, pady=5)
        for text, cmd in [
            ("Snapshot", self.snapshot), ("Record", self.toggle_record),
            ("Load Pattern", self.load_pattern), ("OCR", self.ocr),
            ("QR Decode", self.qr_decode), ("DMC Decode", self.dmc),
            ("Reset WB", self.reset_wb), ("Calibrate WB", self.start_calibrate_wb),
            ("Reset All", self.reset_all), ("Quit", self.on_closing)
        ]:
            ttk.Button(action, text=text, command=cmd).pack(side=tk.LEFT)

        # Mode selectors
        mode_frame = ttk.LabelFrame(self.root, text="Processing Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        for text, val in self.modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.processing_mode, value=val).pack(side=tk.LEFT, padx=2)

        # Parameter sliders
        params = ttk.LabelFrame(self.root, text="Parameters")
        params.pack(fill=tk.X, padx=10, pady=5)
        self.s_edge_low = tk.Scale(params, from_=0, to=255, orient='horizontal', label='Canny Low', command=self.update_edge)
        self.s_edge_low.set(self.params['edge'][0]); self.s_edge_low.pack(side=tk.LEFT)
        self.s_edge_high = tk.Scale(params, from_=0, to=255, orient='horizontal', label='Canny High', command=self.update_edge)
        self.s_edge_high.set(self.params['edge'][1]); self.s_edge_high.pack(side=tk.LEFT)
        self.s_thresh = tk.Scale(params, from_=0, to=255, orient='horizontal', label='Thresh', command=self.update_thresh)
        self.s_thresh.set(self.params['thresh']); self.s_thresh.pack(side=tk.LEFT)
        self.s_pattern = tk.Scale(params, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', label='Pattern', command=self.update_pattern)
        self.s_pattern.set(self.params['pattern_thresh']); self.s_pattern.pack(side=tk.LEFT)
        self.s_blob = tk.Scale(params, from_=0, to=255, orient='horizontal', label='Blob Thresh', command=self.update_blob_params)
        self.s_blob.set(self.params['blob_thresh']); self.s_blob.pack(side=tk.LEFT)
        self.s_blob_min = tk.Scale(params, from_=1, to=10000, orient='horizontal', label='Blob Min Area', command=self.update_blob_params)
        self.s_blob_min.set(self.blob_params['minArea']); self.s_blob_min.pack(side=tk.LEFT)
        self.s_blob_max = tk.Scale(params, from_=1, to=20000, orient='horizontal', label='Blob Max Area', command=self.update_blob_params)
        self.s_blob_max.set(self.blob_params['maxArea']); self.s_blob_max.pack(side=tk.LEFT)
        self.s_blob_circ = tk.Scale(params, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', label='Blob Circ', command=self.update_blob_params)
        self.s_blob_circ.set(self.blob_params['minCircularity']); self.s_blob_circ.pack(side=tk.LEFT)
        self.s_blob_iner = tk.Scale(params, from_=0.0, to=1.0, resolution=0.01, orient='horizontal', label='Blob Inertia', command=self.update_blob_params)
        self.s_blob_iner.set(self.blob_params['minInertiaRatio']); self.s_blob_iner.pack(side=tk.LEFT)

        self.log_widget = scrolledtext.ScrolledText(self.root, height=6, state='disabled')
        self.log_widget.pack(fill=tk.BOTH, padx=10, pady=5)

    def make_blob_detector(self):
        p = cv2.SimpleBlobDetector_Params()
        p.minThreshold = 0
        p.maxThreshold = 255
        p.filterByArea = True
        p.minArea = self.blob_params['minArea']
        p.maxArea = self.blob_params['maxArea']
        p.filterByCircularity = True
        p.minCircularity = self.blob_params['minCircularity']
        p.filterByInertia = True
        p.minInertiaRatio = self.blob_params['minInertiaRatio']
        return cv2.SimpleBlobDetector_create(p)
    
    def process_and_display(self, frame):
        # 1) White balance & grayscale
        img = (frame.astype(np.float32) * self.wb_gain).clip(0,255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mode = self.processing_mode.get()

        # 2) Processing modes
        out = img.copy()
        if mode == 'EdgeDetection':
            edges = cv2.Canny(gray, *self.params['edge'])
            out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == 'Thresholding':
            _,th = cv2.threshold(gray, self.params['thresh'], 255, cv2.THRESH_BINARY)
            out = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        elif mode == 'PatternMatching' and self.template is not None:
            resm = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
            locs = np.where(resm >= self.params['pattern_thresh'])
            for pt in zip(*locs[::-1]):
                cv2.rectangle(out, pt, (pt[0]+self.tpl_size[0], pt[1]+self.tpl_size[1]), (0,255,0), 2)
        elif mode == 'ObjectTracking' and self.tracker is not None:
            ok,box = self.tracker.update(img)
            if ok:
                x,y,w,h = map(int, box)
                cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
        elif mode == 'LineDetection':
            edges = cv2.Canny(gray,50,150)
            lines = cv2.HoughLinesP(edges,1,math.pi/180,50, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for l in lines:
                    x1,y1,x2,y2 = l[0]
                    cv2.line(out, (x1,y1), (x2,y2), (255,0,0), 2)
        elif mode == 'Blurring':
            out = cv2.GaussianBlur(out, (15,15), 0)
        elif mode == 'HistogramEqualization':
            eq = cv2.equalizeHist(gray)
            out = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        elif mode == 'ContourDetection':
            _,thc = cv2.threshold(gray, self.params['thresh'], 255, cv2.THRESH_BINARY)
            cnts,_ = cv2.findContours(thc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0,255,0), 2)
        elif mode == 'BlobDetection':
            kps = self.blob_detector.detect(gray)
            out = cv2.drawKeypoints(out, kps, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 3) Overlays
        self.frame_count += 1
        elapsed = time.time() - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = time.time()
        cv2.putText(out, f"FPS: {self.fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(out, f"Res: {self.current_width}x{self.current_height}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(out, f"Mode: {mode}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Display
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((800,600), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(pil)
        self.video_panel.config(image=tkimg)
        self.video_panel.image = tkimg
        self.last_frame = out

        # Recording
        if self.recording and self.writer is not None:
            self.writer.write(out)

    def grab_loop(self):
        # Throttle to ~30 FPS
        now = time.time()
        if now - self.display_last >= self.display_fps_interval:
            self.display_last = now
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                frame = None
            if frame is not None:
                self.process_and_display(frame)
        self.root.after(1, self.grab_loop)

    def _grab_thread(self):
        while getattr(self, 'camera', None) and self.camera.IsGrabbing():
            try:
                res = self.camera.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    arr = self.converter.Convert(res).GetArray()
                    if not self.frame_queue.empty():
                        try: self.frame_queue.get_nowait()
                        except queue.Empty: pass
                    self.frame_queue.put(arr)
                res.Release()
            except Exception:
                pass

    # Async OCR/QR/DMC
    def ocr(self):
        if hasattr(self,'last_frame'):
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY).copy()
            self.executor.submit(self._do_ocr, gray)

    def _do_ocr(self, gray_frame):
        txt = pytesseract.image_to_string(gray_frame).strip()
        self.log_message(f"OCR: {txt}")

    def qr_decode(self):
        if hasattr(self,'last_frame'):
            frame = self.last_frame.copy()
            self.executor.submit(self._do_qr, frame)

    def _do_qr(self, frame):
        data,_,_ = cv2.QRCodeDetector().detectAndDecode(frame)
        self.log_message(f"QR: {data or 'None'}")

    def dmc(self):
        if hasattr(self,'last_frame'):
            gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY).copy()
            self.executor.submit(self._do_dmc, gray)

    def _do_dmc(self, gray_frame):
        res = decode(gray_frame)
        if res:
            for r in res:
                self.log_message(f"DMC: {r.data.decode()}")
        else:
            self.log_message("DMC: None")

    # Rest of methods
    def on_mouse_down(self, event):
        if self.calibrating and hasattr(self,'last_frame'):
            dw,dh=self.video_panel.winfo_width(),self.video_panel.winfo_height()
            sx=int(event.x*self.current_width/dw); sy=int(event.y*self.current_height/dh)
            b,g,r=self.last_frame[sy,sx].astype(np.float32)
            gains=np.array([255/b if b>0 else 1,255/g if g>0 else 1,255/r if r>0 else 1],dtype=np.float32)
            self.wb_gain=gains; self.calibrating=False
            self.log_message(f"WB calibrated at ({sx},{sy}) BGR=({b:.0f},{g:.0f},{r:.0f})")
            return
        if self.processing_mode.get()=='ObjectTracking':
            self.is_selecting_roi=True; dw,dh=self.video_panel.winfo_width(),self.video_panel.winfo_height()
            sx=int(event.x*self.current_width/dw); sy=int(event.y*self.current_height/dh)
            self.start_x,self.start_y=sx,sy; self.bbox=None; self.log_message("Select ROI for tracking.")

    def on_mouse_drag(self, event):
        if self.is_selecting_roi and self.processing_mode.get()=='ObjectTracking':
            dw,dh=self.video_panel.winfo_width(),self.video_panel.winfo_height()
            cx=int(event.x*self.current_width/dw); cy=int(event.y*self.current_height/dh)
            x,y=min(self.start_x,cx),min(self.start_y,cy)
            w,h=abs(self.start_x-cx),abs(self.start_y-cy)
            self.bbox=(x,y,w,h)

    def on_mouse_up(self, event):
        if self.is_selecting_roi and self.processing_mode.get()=='ObjectTracking':
            self.is_selecting_roi=False
            if self.bbox and self.bbox[2]>0 and self.bbox[3]>0:
                self.tracker=cv2.TrackerCSRT_create(); self.tracker.init(self.last_frame,self.bbox)
                self.log_message(f"Tracker initialized on {self.bbox}")
            else:
                self.log_message("Invalid ROI selected."); self.bbox=None

    def connect_camera(self):
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_camera_open = True
            self.log_message("Camera connected.")
        except Exception as e:
            self.is_camera_open = False
            self.log_message(f"Camera connection failed: {e}")

    def update_blob_params(self, _=None):
        self.params['blob_thresh'] = self.s_blob.get()
        self.blob_params['minArea'] = self.s_blob_min.get()
        self.blob_params['maxArea'] = self.s_blob_max.get()
        self.blob_params['minCircularity'] = self.s_blob_circ.get()
        self.blob_params['minInertiaRatio'] = self.s_blob_iner.get()
        self.blob_detector = self.make_blob_detector()

    def start_calibrate_wb(self):
        self.calibrating = True
        self.log_message("Click on a white region in the video to calibrate white balance.")

    def setup_resolutions(self):
        max_w = self.camera.Width.GetMax()
        max_h = self.camera.Height.GetMax()
        g = math.gcd(max_w, max_h)
        rw, rh = max_w // g, max_h // g
        candidates = [(rw * k, rh * k) for k in range(g, 0, -1)]
        supported = [(w, h) for w, h in candidates if w % 2 == 0 and h % 2 == 0]
        vals = [f"{w}x{h}" for w, h in supported]
        self.res_combo['values'] = vals
        self.res_var.set(f"{self.current_width}x{self.current_height}")

    def apply_resolution(self):
        sel = self.res_var.get()
        if not sel:
            self.log_message("Select a resolution first.")
            return
        w, h = map(int, sel.split('x'))
        self.set_camera_resolution(w, h)

    def apply_custom_resolution(self):
        try:
            w, h = int(self.w_entry.get()), int(self.h_entry.get())
        except ValueError:
            self.log_message("Invalid custom resolution values.")
            return
        self.set_camera_resolution(w, h)

    def set_camera_resolution(self, w, h):
        try:
            if self.camera.IsGrabbing(): self.camera.StopGrabbing()
            self.camera.Width.SetValue(w); self.camera.Height.SetValue(h)
            self.current_width, self.current_height = w, h
            self.log_message(f"Resolution set to {w}x{h}")
        except Exception as e:
            self.log_message(f"Failed to set resolution: {e}")
        finally:
            if not self.camera.IsGrabbing(): self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def update_edge(self, _=None): self.params['edge'] = (self.s_edge_low.get(), self.s_edge_high.get())
    def update_thresh(self, _=None): self.params['thresh'] = self.s_thresh.get()
    def update_pattern(self, _=None): self.params['pattern_thresh'] = self.s_pattern.get()

    def load_pattern(self):
        fn = filedialog.askopenfilename(filetypes=[('Image Files','*.png;*.jpg;*.jpeg;*.bmp')])
        if fn:
            tpl = cv2.imread(fn,0); self.template = tpl; self.tpl_size = tpl.shape[::-1]
            self.log_message(f"Loaded pattern size {self.tpl_size}")
            self.processing_mode.set("PatternMatching")

    def reset_wb(self): self.wb_gain = np.ones(3, dtype=np.float32); self.log_message("White balance reset.")
    def reset_all(self):
        self.reset_wb(); self.s_edge_low.set(50); self.s_edge_high.set(150)
        self.s_thresh.set(128); self.s_pattern.set(0.7); self.s_blob.set(128)
        self.processing_mode.set("None"); self.template=None; self.tracker=None; self.bbox=None
        self.log_message("All parameters reset.")

    def snapshot(self):
        if hasattr(self,'last_frame'):
            fn = filedialog.asksaveasfilename(defaultextension='.png'); cv2.imwrite(fn, self.last_frame)
            self.log_message(f"Snapshot saved: {fn}")

    def toggle_record(self):
        if not self.recording:
            fn = filedialog.asksaveasfilename(defaultextension='.avi')
            if fn: 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer=cv2.VideoWriter(fn,fourcc,20,(self.current_width,self.current_height))
                self.recording=True; self.log_message("Recording started.")
        else:
            self.recording=False; self.writer.release(); self.log_message("Recording stopped.")

    def log_message(self,msg):
        self.log_widget['state']='normal'; self.log_widget.insert(tk.END,msg+"\n"); self.log_widget.see(tk.END); self.log_widget['state']='disabled'

    def on_closing(self):
        self.log_message("Exiting...")
        if hasattr(self,'camera') and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
            self.camera.Close()
        if self.recording and self.writer:
            self.writer.release()
        self.root.destroy()

if __name__ == '__main__':
    root=tk.Tk()
    app=VisionCameraApp(root)
    root.mainloop()