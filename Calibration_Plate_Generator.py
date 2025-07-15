import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
from PIL import Image, ImageTk

class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calibration Plate Generator")
        self.resizable(False, False)
        self._init_vars()
        self._create_widgets()
        self._update_param_frames()

    def _init_vars(self):
        # Common
        self.pattern_var = tk.StringVar(value="Chessboard")
        self.outline_var = tk.BooleanVar(value=False)
        self.border_thickness = tk.IntVar(value=1)
        self.outdir = tk.StringVar(value=os.getcwd())
        self.filename = tk.StringVar(value="pattern.png")
        self.format_var = tk.StringVar(value="PNG")
        self.dpi = tk.IntVar(value=300)
        self.batch_count = tk.IntVar(value=1)
        self.auto_increment = tk.BooleanVar(value=False)
        # Chessboard
        self.cols = tk.IntVar(value=9)
        self.rows = tk.IntVar(value=6)
        self.square_size = tk.IntVar(value=50)
        # Circles
        self.spacing = tk.IntVar(value=100)
        self.radius = tk.IntVar(value=20)

    def _create_widgets(self):
        padx, pady = 5, 5
        main = ttk.Frame(self, padding=(10,10))
        main.grid()
        # Pattern selection
        ttk.Label(main, text="Pattern:").grid(row=0, column=0, sticky="e", padx=padx, pady=pady)
        cb = ttk.Combobox(main, textvariable=self.pattern_var, values=["Chessboard","Circles"], state="readonly", width=12)
        cb.grid(row=0, column=1, sticky="w", padx=padx, pady=pady)
        cb.bind("<<ComboboxSelected>>", lambda e: self._update_param_frames())
        # Parameter frames
        self.cb_frame = ttk.LabelFrame(main, text="Chessboard Params")
        self.cb_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=padx, pady=pady)
        ttk.Label(self.cb_frame, text="Cols:").grid(row=0, column=0, padx=padx)
        ttk.Entry(self.cb_frame, textvariable=self.cols, width=5).grid(row=0, column=1)
        ttk.Label(self.cb_frame, text="Rows:").grid(row=0, column=2)
        ttk.Entry(self.cb_frame, textvariable=self.rows, width=5).grid(row=0, column=3)
        ttk.Label(self.cb_frame, text="Square Size:").grid(row=1, column=0)
        ttk.Entry(self.cb_frame, textvariable=self.square_size, width=5).grid(row=1, column=1)

        self.cir_frame = ttk.LabelFrame(main, text="Circles Params")
        self.cir_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=padx, pady=pady)
        ttk.Label(self.cir_frame, text="Spacing:").grid(row=0, column=0, padx=padx)
        ttk.Entry(self.cir_frame, textvariable=self.spacing, width=5).grid(row=0, column=1)
        ttk.Label(self.cir_frame, text="Radius:").grid(row=0, column=2)
        ttk.Entry(self.cir_frame, textvariable=self.radius, width=5).grid(row=0, column=3)

        # Outline and thickness
        ttk.Checkbutton(main, text="Outline Only", variable=self.outline_var).grid(row=3, column=0, padx=padx, pady=pady)
        ttk.Label(main, text="Border Thickness:").grid(row=3, column=1)
        ttk.Entry(main, textvariable=self.border_thickness, width=5).grid(row=3, column=2)

        # Output settings
        output = ttk.LabelFrame(main, text="Output")
        output.grid(row=4, column=0, columnspan=3, sticky="ew", padx=padx, pady=pady)
        ttk.Label(output, text="Folder:").grid(row=0, column=0, padx=padx)
        ttk.Entry(output, textvariable=self.outdir, width=25).grid(row=0, column=1)
        ttk.Button(output, text="Browse", command=self._browse).grid(row=0, column=2, padx=padx)
        ttk.Label(output, text="Filename:").grid(row=1, column=0)
        ttk.Entry(output, textvariable=self.filename, width=25).grid(row=1, column=1, columnspan=2)
        ttk.Label(output, text="Format:").grid(row=2, column=0)
        ttk.Combobox(output, textvariable=self.format_var, values=["PNG","JPEG","TIFF"], state="readonly", width=5).grid(row=2, column=1)
        ttk.Label(output, text="DPI:").grid(row=2, column=2)
        ttk.Entry(output, textvariable=self.dpi, width=5).grid(row=2, column=3)
        ttk.Label(output, text="Batch Count:").grid(row=3, column=0)
        ttk.Entry(output, textvariable=self.batch_count, width=5).grid(row=3, column=1)
        ttk.Checkbutton(output, text="Auto-Increment", variable=self.auto_increment).grid(row=3, column=2, columnspan=2)

        # Actions
        ttk.Button(main, text="Preview", command=self._preview).grid(row=5, column=0, padx=padx, pady=pady)
        ttk.Button(main, text="Generate", command=self._generate).grid(row=5, column=1)
        # Preview area
        self.preview_lbl = ttk.Label(main)
        self.preview_lbl.grid(row=6, column=0, columnspan=3, pady=(10,0))

    def _update_param_frames(self):
        if self.pattern_var.get() == "Chessboard":
            self.cb_frame.lift()
            self.cir_frame.lower()
        else:
            self.cir_frame.lift()
            self.cb_frame.lower()

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get())
        if d:
            self.outdir.set(d)

    def _make_pattern(self):
        outline = self.outline_var.get()
        thickness = self.border_thickness.get()
        if self.pattern_var.get() == "Chessboard":
            cols, rows, sz = self.cols.get(), self.rows.get(), self.square_size.get()
            img = 255*np.ones((rows*sz, cols*sz),dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    x,y = c*sz, r*sz
                    if outline:
                        cv2.rectangle(img,(x,y),(x+sz,y+sz),0,thickness)
                    elif (r+c)%2==0:
                        cv2.rectangle(img,(x,y),(x+sz,y+sz),0,-1)
        else:
            cols, rows, sp, rad = self.cols.get(), self.rows.get(), self.spacing.get(), self.radius.get()
            w = (cols-1)*sp+2*rad; h = (rows-1)*sp+2*rad
            img = 255*np.ones((h,w,3),dtype=np.uint8)
            for r in range(rows):
                for c in range(cols):
                    cx, cy = rad+c*sp, rad+r*sp
                    if outline:
                        cv2.circle(img,(cx,cy),rad,(0,0,0),thickness)
                    else:
                        cv2.circle(img,(cx,cy),rad,(0,0,0),-1)
        return img

    def _preview(self):
        try:
            img = self._make_pattern()
            disp = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) if img.ndim==2 else cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(disp); pil.thumbnail((300,300))
            self.tkimg = ImageTk.PhotoImage(pil)
            self.preview_lbl.config(image=self.tkimg)
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def _generate(self):
        try:
            img = self._make_pattern()
            base, ext = os.path.splitext(self.filename.get())
            for i in range(self.batch_count.get()):
                name = f"{base}_{i+1}{ext}" if self.auto_increment.get() else base+ext
                path = os.path.join(self.outdir.get(), name)
                if img.ndim==2:
                    pil = Image.fromarray(img)
                else:
                    pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                pil.save(path, format=self.format_var.get(), dpi=(self.dpi.get(),self.dpi.get()))
            messagebox.showinfo("Saved",f"Saved {self.batch_count.get()} files to {self.outdir.get()}")
        except Exception as e:
            messagebox.showerror("Error",str(e))

if __name__ == '__main__':
    CalibrationApp().mainloop()