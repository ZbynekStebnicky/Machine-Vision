import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pypylon import pylon

# --- PARAMETERS ---
CHECKER_COLS = 9        # number of squares width in chessboard pattern
CHECKER_ROWS = 6        # number of squares height in chessboard pattern
SQUARE_SIZE_MM = 25.0   # size of each square in millimeters
CALIB_FILE = 'calibration.npz'    # file to store intrinsic calibration data
EXTR_FILE = 'extrinsics.npz'      # file to store extrinsic calibration data

class BaslerCam:
    """
    Wrapper for Basler camera using pypylon.
    Handles opening, grabbing frames, and conversion to OpenCV format.
    """
    def __init__(self):
        # Initialize camera device and set up buffer and converter
        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()
        self.cam.MaxNumBuffer = 5
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def get_frame(self):
        """
        Grab a single frame within a timeout (ms).
        Returns an OpenCV BGR numpy array or None if failed.
        """
        res = self.cam.GrabOne(1000)
        if res.GrabSucceeded():
            img = self.conv.Convert(res).GetArray()
            return img
        return None

    def close(self):
        """
        Close the camera device cleanly.
        """
        self.cam.Close()

class App(tk.Tk):
    """
    Main application GUI for camera calibration (intrinsics, extrinsics)
    and tracking via optical flow.
    """
    def __init__(self):
        super().__init__()
        # Configure main window
        self.geometry('1500x900')
        self.title('Calibration & Tracking GUI')
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Setup Basler camera
        self.camera = BaslerCam()

        # Calibration flags
        self.intr_mode = False    # Intrinsic calibration mode
        self.extr_mode = False    # Extrinsic calibration mode

        # Prepare object points for chessboard (3D world points)
        self.objp = np.zeros(((CHECKER_COLS-1)*(CHECKER_ROWS-1),3), np.float32)
        self.objp[:,:2] = np.mgrid[0:CHECKER_COLS-1, 0:CHECKER_ROWS-1].T.reshape(-1,2)
        self.objp *= SQUARE_SIZE_MM
        # Lists to store object points and image points
        self.objpoints = []
        self.imgpoints = []

        # Camera parameters and pose
        self.cam_mtx = None
        self.dist = None
        self.rvec = None
        self.tvec = None

        # Build UI elements
        self.lbl = tk.Label(self)
        self.lbl.config(width=1100, height=800)
        self.lbl.pack()

        # Control buttons
        frm = tk.Frame(self)
        frm.pack(fill='x')
        tk.Button(frm, text='Start Intrinsics', command=self.start_intr).pack(side='left')
        tk.Button(frm, text='Capture Frame', command=self.capture_frame).pack(side='left')
        tk.Button(frm, text='Finish Intrinsics', command=self.finish_intr).pack(side='left')
        tk.Button(frm, text='Start Extrinsics', command=self.start_extr).pack(side='left')
        tk.Button(frm, text='Finish Extrinsics', command=self.finish_extr).pack(side='left')
        tk.Button(frm, text='Select & Track', command=self.start_track).pack(side='left')
        
        # Status and numeric display
        self.info = tk.Label(self, text='Ready')
        self.info.pack()
        self.dx_label = tk.Label(self, text='Δx: 0.0 mm')
        self.dx_label.pack()
        self.dy_label = tk.Label(self, text='Δy: 0.0 mm')
        self.dy_label.pack()

        # Tracking variables
        self.selected = False       # Tracking started
        self.init_pt = None         # Initial pixel location
        self.prev_gray = None       # Previous gray frame for optical flow
        self.init_world = None      # Initial world coordinates (X, Y)

        # Start periodic update loop
        self.after(30, self.update)

    def update(self):
        """
        Main loop: grab frame, overlay calibration or tracking info,
        and update display.
        """
        frame = self.camera.get_frame()
        if frame is not None:
            # Display at original resolution
            vis = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Overlay chessboard corners if calibrating
            if self.intr_mode or self.extr_mode:
                ret, corners = cv2.findChessboardCorners(
                    gray, (CHECKER_COLS-1, CHECKER_ROWS-1)
                )
                if ret:
                    cv2.drawChessboardCorners(
                        vis, (CHECKER_COLS-1, CHECKER_ROWS-1), corners, ret
                    )

            # If tracking enabled, compute optical flow and draw arrow
            if self.selected:
                nextPts, st, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.init_pt, None
                )
                if st[0][0] == 1:
                    curr = nextPts
                    x0, y0 = self.init_pt[0]
                    x1, y1 = curr[0]
                    # Draw arrow from previous to current point
                    cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0,255,0), 2)

                    # Convert pixel motion to world (mm)
                    X1, Y1 = self.pix2world(x1, y1)
                    dx = X1 - self.init_world[0]
                    dy = Y1 - self.init_world[1]
                    # Draw numeric overlay
                    cv2.putText(vis, f"Δx={dx:.1f}mm Δy={dy:.1f}mm",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # Update labels and previous state
                    self.dx_label.config(text=f'Δx: {dx:.1f} mm')
                    self.dy_label.config(text=f'Δy: {dy:.1f} mm')
                    self.prev_gray = gray.copy()
                    self.init_pt = curr
                else:
                    self.info.config(text='Tracking lost')

            # Convert and display in Tkinter
            img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            self.lbl.imgtk = imgtk
            self.lbl.config(image=imgtk)
            self.latest_gray = gray

        # Schedule next update
        self.after(30, self.update)

    def start_intr(self):
        """
        Begin collecting frames for intrinsic calibration.
        """
        self.intr_mode = True
        self.objpoints.clear()
        self.imgpoints.clear()
        self.info.config(text='Intrinsics: click Capture to save frames')

    def capture_frame(self):
        """
        Capture detected chessboard corners from latest frame
        and append to calibration data.
        """
        if not self.intr_mode:
            return
        gray = self.latest_gray
        ret, corners = cv2.findChessboardCorners(
            gray, (CHECKER_COLS-1, CHECKER_ROWS-1)
        )
        if ret:
            self.objpoints.append(self.objp.copy())
            self.imgpoints.append(corners.copy())
            self.info.config(text=f'Captured {len(self.objpoints)} frames')
        else:
            self.info.config(text='No chessboard detected')

    def finish_intr(self):
        """
        Compute intrinsic camera matrix and distortion coefficients
        once enough frames are captured.
        """
        if not self.intr_mode:
            return
        if len(self.objpoints) < 10:
            messagebox.showerror('Error', 'Need ≥10 frames')
            return
        h, w = self.latest_gray.shape
        rms, self.cam_mtx, self.dist, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w,h), None, None
        )
        # Save calibration to file
        np.savez(CALIB_FILE, camera_matrix=self.cam_mtx, dist_coeffs=self.dist)
        self.intr_mode = False
        self.info.config(text=f'Intrinsics done RMS={rms:.3f}')

    def start_extr(self):
        """
        Load intrinsic parameters (if needed) and start extrinsic pose calibration.
        """
        if self.cam_mtx is None:
            data = np.load(CALIB_FILE)
            self.cam_mtx, self.dist = data['camera_matrix'], data['dist_coeffs']
        self.extr_mode = True
        self.info.config(text='Extrinsics: position and press Finish')

    def finish_extr(self):
        """
        Solve for camera rotation and translation based on one chessboard frame.
        """
        if not self.extr_mode:
            return
        gray = self.latest_gray
        ret, corners = cv2.findChessboardCorners(
            gray, (CHECKER_COLS-1, CHECKER_ROWS-1)
        )
        if not ret:
            messagebox.showerror('Error', 'Chessboard not found')
            return
        _, rvec, tvec = cv2.solvePnP(
            self.objp, corners, self.cam_mtx, self.dist
        )
        self.rvec, self.tvec = rvec, tvec
        np.savez(EXTR_FILE, rvec=rvec, tvec=tvec)
        self.extr_mode = False
        self.info.config(text='Extrinsics done')

    def pix2world(self, u, v):
        """
        Convert pixel coordinates (u,v) to world coordinates (X,Y) in mm
        assuming plane Z=0 (chessboard plane).
        """
        # Create normalized image ray
        uv1 = np.array([u, v, 1.0])
        inv = np.linalg.inv(self.cam_mtx)
        ray = inv.dot(uv1)
        # Invert rotation and translation to transform ray into world
        R, _ = cv2.Rodrigues(self.rvec)
        Rinv = R.T
        t = self.tvec.reshape(3,1)
        # Scale factor so that point lies on Z=0 plane
        s = (Rinv.dot(t))[2] / (Rinv.dot(ray.reshape(3,1)))[2]
        campt = (ray * s).reshape(3,1)
        wpt = Rinv.dot(campt - t)
        return float(wpt[0][0]), float(wpt[1][0])

    def start_track(self):
        """
        Enable user to click an object in the view to begin tracking.
        """
        if self.rvec is None:
            messagebox.showerror('Error', 'Calibrate extrinsics first')
            return
        self.info.config(text='Click on object to track')
        self.lbl.bind('<Button-1>', self.select_pt)

    def select_pt(self, event):
        """
        Handle mouse click: record initial pixel and world coordinates
        and start optical flow tracking.
        """
        x, y = event.x, event.y
        # Map GUI click to image coordinates
        h, w = self.latest_gray.shape
        fw, fh = self.lbl.winfo_width(), self.lbl.winfo_height()
        u = x * w / fw
        v = y * h / fh
        self.init_pt = np.array([[u, v]], np.float32)
        self.prev_gray = self.latest_gray.copy()
        X, Y = self.pix2world(u, v)
        self.init_world = (X, Y)
        self.selected = True
        self.info.config(text=f'Initial ({X:.1f}mm,{Y:.1f}mm)')
        # Unbind to prevent multiple clicks
        self.lbl.unbind('<Button-1>')

    def on_close(self):
        """
        Clean up camera and close the application.
        """
        self.camera.close()
        self.destroy()

if __name__ == '__main__':
    # Launch the GUI application
    App().mainloop()