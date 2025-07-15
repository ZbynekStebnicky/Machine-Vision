import cv2
import numpy as np
import threading
import time
import os
from pypylon import pylon
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ------- PARAMETERS & LANGUAGES -------
LANGUAGES = {
    "EN": {
        "save_frame": "Save Current Frame",
        "auto_capture": "Auto-capture",
        "calibrate": "Calibrate Camera",
        "verify": "Verify Calibration",
        "undistort_img": "Undistort Image",
        "gallery": "Frame Gallery",
        "settings": "Settings",
        "exposure": "Exposure",
        "quit": "Quit",
        "about": "Help/About",
        "detected": "Chessboard detected!",
        "not_detected": "No chessboard",
        "calib_ok": "Calibration complete.\nRMS error: {rms:.4f}",
        "calib_params": "Camera Matrix:\n{cm}\n\nDistortion:\n{dc}",
        "error": "Error",
        "info": "Info",
        "lang": "Language"
    },
    "CZ": {
        "save_frame": "Uložit snímek",
        "auto_capture": "Automatické ukládání",
        "calibrate": "Kalibrovat kameru",
        "verify": "Ověřit kalibraci",
        "undistort_img": "Narovnat obrázek",
        "gallery": "Galerie snímků",
        "settings": "Nastavení",
        "exposure": "Expozice",
        "quit": "Konec",
        "about": "Nápověda/O aplikaci",
        "detected": "Šachovnice detekována!",
        "not_detected": "Šachovnice nenalezena",
        "calib_ok": "Kalibrace dokončena.\nRMS chyba: {rms:.4f}",
        "calib_params": "Matice kamery:\n{cm}\n\nZkreslení:\n{dc}",
        "error": "Chyba",
        "info": "Info",
        "lang": "Jazyk"
    }
}
current_lang = "EN"
L = LANGUAGES[current_lang]

# ------- User-selectable parameters -------
def get_params_dialog():
    global CHECKERBOARD, square_size
    root = tk.Tk()
    root.withdraw()
    cols = simpledialog.askinteger("Checkerboard", "Number of squares (columns, e.g. 9):", initialvalue=9)
    rows = simpledialog.askinteger("Checkerboard", "Number of squares (rows, e.g. 6):", initialvalue=6)
    squaresize = simpledialog.askfloat("Checkerboard", "Square size (mm):", initialvalue=25.0)
    root.destroy()
    if cols is None or rows is None or squaresize is None:
        cols, rows, squaresize = 9, 6, 25.0
    # checkerboard = (inner_corners_w, inner_corners_h)
    CHECKERBOARD = (cols-1, rows-1)
    return CHECKERBOARD, squaresize

CHECKERBOARD, square_size = get_params_dialog()
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# ------- Calibration session storage -------
SESSION_DIR = "calib_session"
os.makedirs(SESSION_DIR, exist_ok=True)
CAPTURED_IMAGES = []  # (filename, imgpoints)

# ------- Globals -------
objpoints = []
imgpoints = []
latest_gray = None
latest_frame = None
latest_corners = None
found_corners = False
running = True
auto_capture = False
gallery_update_needed = False
fps = 0
exposure = 20000  # in us

# ------- Basler camera setup -------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.MaxNumBuffer = 5
try:
    camera.Width.Value = 1260
    camera.Height.Value = 940
    #camera.ExposureTime.Value = exposure
except Exception as e:
    print(e)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_RGB8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# ------- GUI ---------
class CalibGUI:
    def __init__(self, root):
        global L
        self.root = root
        self.root.title("Basler Camera Calibration (PRO)")
        self.status_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.status_label.pack(pady=3)
        self.fps_label = tk.Label(root, text="FPS: 0", font=("Arial", 10))
        self.fps_label.pack()
        self.preview_panel = tk.Label(root)
        self.preview_panel.pack()

        # Controls
        frm = tk.Frame(root)
        frm.pack()
        self.save_btn = tk.Button(frm, text=L["save_frame"], width=20, command=self.save_frame)
        self.save_btn.grid(row=0, column=0, padx=3, pady=2)
        self.auto_var = tk.BooleanVar()
        self.auto_chk = tk.Checkbutton(frm, text=L["auto_capture"], variable=self.auto_var, command=self.toggle_auto)
        self.auto_chk.grid(row=0, column=1)
        self.calib_btn = tk.Button(frm, text=L["calibrate"], width=20, command=self.calibrate)
        self.calib_btn.grid(row=1, column=0, padx=3, pady=2)
        self.verify_btn = tk.Button(frm, text=L["verify"], width=20, command=self.verify_calibration)
        self.verify_btn.grid(row=1, column=1, padx=3, pady=2)
        self.undist_btn = tk.Button(frm, text=L["undistort_img"], width=20, command=self.undistort_image)
        self.undist_btn.grid(row=2, column=0, padx=3, pady=2)
        self.gallery_btn = tk.Button(frm, text=L["gallery"], width=20, command=self.show_gallery)
        self.gallery_btn.grid(row=2, column=1, padx=3, pady=2)
        self.settings_btn = tk.Button(frm, text=L["settings"], width=20, command=self.show_settings)
        self.settings_btn.grid(row=3, column=0, padx=3, pady=2)
        self.about_btn = tk.Button(frm, text=L["about"], width=20, command=self.show_about)
        self.about_btn.grid(row=3, column=1, padx=3, pady=2)
        self.quit_btn = tk.Button(frm, text=L["quit"], width=20, command=self.quit_app)
        self.quit_btn.grid(row=4, column=0, columnspan=2, pady=10)

        # Exposure slider
        #self.exp_slider = ttk.Scale(root, from_=100, to=80000, orient="horizontal", command=self.set_exposure)
        #self.exp_slider.set(exposure)
        #tk.Label(root, text=L["exposure"]).pack()
        #self.exp_slider.pack(fill="x", padx=20)

        # Language switch
        self.lang_var = tk.StringVar(value=current_lang)
        self.lang_menu = ttk.OptionMenu(root, self.lang_var, current_lang, *LANGUAGES.keys(), command=self.set_language)
        self.lang_menu.pack(pady=4)

        # Drag-and-drop undistort
        if DND_AVAILABLE:
            self.root.drop_target_register('*')
            self.root.dnd_bind('<<Drop>>', self.dnd_undistort)

        self.last_auto_save = time.time()
        self.preview_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.preview_thread.start()

        self.update_gui()

    def set_language(self, lang):
        global L, current_lang
        current_lang = lang
        L = LANGUAGES[lang]
        # Just reload everything (quick way)
        self.root.destroy()
        main()

    #def set_exposure(self, val):
    #    global exposure
    #    try:
    #        exposure = int(float(val))
    #        camera.ExposureTime.Value = exposure
    #    except Exception as e:
    #        pass

    def update_gui(self):
        global fps
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.root.after(500, self.update_gui)

    def toggle_auto(self):
        global auto_capture
        auto_capture = self.auto_var.get()

    def camera_loop(self):
        global latest_frame, latest_gray, latest_corners, found_corners, fps, objpoints, imgpoints, CAPTURED_IMAGES, gallery_update_needed
        last = time.time()
        frames = 0
        while running:
            grab_result = camera.GrabOne(1000)
            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                frame = image.GetArray()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
                display_frame = frame.copy()

                if ret:
                    cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret)
                    self.status_label.config(text=L["detected"], fg="green")
                    latest_corners = corners
                    found_corners = True
                    if auto_capture and (time.time()-self.last_auto_save > 1.5):
                        # Save frame
                        fname = os.path.join(SESSION_DIR, f"img_{len(CAPTURED_IMAGES):03d}.png")
                        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        objpoints.append(objp.copy())
                        imgpoints.append(corners.copy())
                        CAPTURED_IMAGES.append((fname, corners.copy()))
                        self.last_auto_save = time.time()
                        gallery_update_needed = True
                else:
                    self.status_label.config(text=L["not_detected"], fg="red")
                    found_corners = False

                # FPS count
                frames += 1
                if frames >= 10:
                    now = time.time()
                    fps = 10 / (now - last)
                    last = now
                    frames = 0

                # Show preview
                disp_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                disp_bgr = cv2.resize(disp_bgr, (640, 480))
                img = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                from PIL import Image, ImageTk
                imgtk = ImageTk.PhotoImage(Image.fromarray(img))
                self.preview_panel.imgtk = imgtk
                self.preview_panel.config(image=imgtk)
                latest_frame = frame
                latest_gray = gray
        camera.Close()

    def save_frame(self):
        if found_corners:
            fname = os.path.join(SESSION_DIR, f"img_{len(CAPTURED_IMAGES):03d}.png")
            cv2.imwrite(fname, cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR))
            objpoints.append(objp.copy())
            imgpoints.append(latest_corners.copy())
            CAPTURED_IMAGES.append((fname, latest_corners.copy()))
            messagebox.showinfo(L["info"], f"Frame saved ({len(objpoints)} total).")
        else:
            messagebox.showwarning(L["error"], L["not_detected"])

    def show_gallery(self):
        g = tk.Toplevel(self.root)
        g.title(L["gallery"])
        for idx, (fname, corners) in enumerate(CAPTURED_IMAGES):
            img = cv2.imread(fname)
            if img is not None:
                img = cv2.resize(img, (120, 90))
                from PIL import Image, ImageTk
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                lbl = tk.Label(g, image=imgtk)
                lbl.image = imgtk
                lbl.grid(row=idx // 6, column=idx % 6)
                btn = tk.Button(g, text="X", command=lambda i=idx: self.delete_gallery_frame(i))
                btn.grid(row=(idx // 6)+1, column=idx % 6)
        g.mainloop()

    def delete_gallery_frame(self, idx):
        # Remove both image and points
        fname, _ = CAPTURED_IMAGES[idx]
        if os.path.exists(fname):
            os.remove(fname)
        del CAPTURED_IMAGES[idx]
        del objpoints[idx]
        del imgpoints[idx]
        messagebox.showinfo(L["info"], f"Deleted frame {idx}. Please reopen gallery.")

    def calibrate(self):
        if len(objpoints) < 10:
            messagebox.showerror(L["error"], "Need at least 10 frames for calibration.")
            return
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, latest_gray.shape[::-1], None, None
        )
        np.savez("basler_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        h, w = latest_gray.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(latest_frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
        undistorted_bgr = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
        cv2.imshow("Undistorted Verification", undistorted_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow("Undistorted Verification")
        msg = L["calib_ok"].format(rms=rms)
        params = L["calib_params"].format(cm=str(camera_matrix), dc=str(dist_coeffs))
        messagebox.showinfo(L["info"], f"{msg}\n\n{params}")

    def verify_calibration(self):
        if not os.path.exists("basler_calibration.npz"):
            messagebox.showerror(L["error"], "Calibration file not found.")
            return
        if latest_frame is None:
            messagebox.showerror(L["error"], "No camera image available.")
            return
        data = np.load("basler_calibration.npz")
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        h, w = latest_frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(latest_frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
        orig_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)
        und_bgr = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
        both = np.hstack([orig_bgr, und_bgr])
        cv2.imshow("Calibration Check (Left=Original, Right=Undistorted)", both)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration Check (Left=Original, Right=Undistorted)")

    def undistort_image(self):
        if not os.path.exists("basler_calibration.npz"):
            messagebox.showerror(L["error"], "Calibration file not found.")
            return
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select image to undistort", filetypes=filetypes)
        if not filename:
            return
        img = cv2.imread(filename)
        data = np.load("basler_calibration.npz")
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
        outname = filedialog.asksaveasfilename(title="Save undistorted image as...", defaultextension=".png")
        if outname:
            cv2.imwrite(outname, undistorted)
            messagebox.showinfo(L["info"], f"Saved: {outname}")

    def dnd_undistort(self, event):
        # Drag and drop for undistortion
        filename = event.data
        if filename and os.path.exists(filename):
            img = cv2.imread(filename)
            if not os.path.exists("basler_calibration.npz"):
                messagebox.showerror(L["error"], "Calibration file not found.")
                return
            data = np.load("basler_calibration.npz")
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
            h, w = img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)
            both = np.hstack([img, undistorted])
            cv2.imshow("Undistorted", both)
            cv2.waitKey(0)
            cv2.destroyWindow("Undistorted")

    def show_settings(self):
        s = tk.Toplevel(self.root)
        s.title(L["settings"])
        tk.Label(s, text=f"Checkerboard: {CHECKERBOARD}\nSquare size: {square_size} mm\nSession dir: {SESSION_DIR}\nExposure: {exposure} us").pack(pady=10)

    def show_about(self):
        messagebox.showinfo(L["about"], "Basler Camera Calibration Tool\nFeatures:\n- Auto/manual capture\n- Frame gallery\n- Live checkerboard detection\n- Detailed calibration\n- Image undistortion tool\n- Multi-language\n- Exposure slider\n- Session save/load\n- Drag-and-drop undistort\n")

    def quit_app(self):
        global running
        running = False
        self.root.destroy()

def main():
    global running
    running = True
    try:
        root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    except Exception:
        root = tk.Tk()
    gui = CalibGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.quit_app)
    root.mainloop()

if __name__ == "__main__":
    main()