import cv2
import numpy as np

def to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, ksize=(5, 5)):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, ksize, 0)

def enhance_contrast(image):
    """Enhance image contrast using CLAHE (adaptive histogram equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image) 