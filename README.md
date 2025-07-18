# Basler Vision Suite

This repository contains a set of Python tools designed for machine vision applications using Basler industrial cameras. It supports real-time image acquisition, camera calibration, image processing, code recognition (OCR, QR, DataMatrix), calibration pattern generation, and CalibTrack tool for combined calibration and object tracking. The solution is built with usability in mind, featuring intuitive graphical user interfaces (GUIs) powered by Tkinter.

## Features
### 1. *Basler_gui.py* – Vision Processing Interface

An interactive GUI for live video streaming from Basler cameras with multiple vision tools:

- Live processing modes: Edge Detection, Thresholding, Contour Detection, Histogram Equalization, etc.

- Object tracking via OpenCV Trackers

- Optical character recognition (OCR) via Tesseract

- QR and DataMatrix code decoding

- Snapshot and video recording capabilities

- Interactive white balance calibration

- Adjustable blob detection parameters

- Dynamic resolution setting and preview

### 2. *Basler_Calibration_gui.py* – Camera Calibration Tool

A comprehensive GUI for calibrating Basler cameras using a chessboard pattern:

- Live checkerboard detection with auto/manual frame capture

- Calibration of intrinsic camera parameters using OpenCV

- Undistortion of images based on saved calibration parameters

- Frame gallery for reviewing captured calibration frames

- Drag-and-drop support for undistortion

- Multilingual support (English/Czech)

### 3. *Calibration_Plate_Generator.py* – Calibration Pattern Generator

Generates printable calibration patterns (chessboard or circular dot grids):

- Customizable size, DPI, and output format

- Option to generate multiple patterns in batch

- Preview function for visual inspection

- Outline-only mode and border thickness setting

### 4. *CalibTrack.py* – Calibration & Tracking GUI

Combines intrinsic/extrinsic calibration with real-time point tracking:

- Intrinsic calibration using chessboard pattern (auto/manual capture)

- Extrinsic pose estimation (solvePnP) and saving of rvec/tvec

- Pixel-to-world conversion on planar calibration surface

- Click-to-select an object and optical flow tracking (LK)

- Real-time display of ΔX/ΔY movement in millimeters

- Save/load of calibration data for repeatable measurements

## Dependencies
Make sure you have the following Python packages installed:
```bash
opencv-python

numpy

pypylon (for Basler camera interface)

Pillow

tkinter

pytesseract

pylibdmtx

tkinterdnd2 (optional, for drag-and-drop support)
```

Tesseract OCR must also be installed separately and its path configured in *Basler_gui.py*.
