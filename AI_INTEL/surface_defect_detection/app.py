import streamlit as st
import os
import tempfile
from src.utils import load_image
from src.preprocess import to_grayscale, apply_gaussian_blur, enhance_contrast
import cv2
import pandas as pd

st.title('Surface Defect Detection on Steel Strips')

method = st.selectbox('Detection Method', ['classical', 'advanced classical'])

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

def advanced_classical_detection(img):
    gray = to_grayscale(img)
    blur = apply_gaussian_blur(gray, ksize=(7, 7))
    contrast = enhance_contrast(blur)
    # Use Canny edge detection
    edges = cv2.Canny(contrast, 50, 150)
    # Morphological closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter by area
    bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
    bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
    return bboxes, closed

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    output_dir = tempfile.mkdtemp()
    img = load_image(tmp_path)
    if method == 'classical':
        gray = to_grayscale(img)
        blur = apply_gaussian_blur(gray)
        contrast = enhance_contrast(blur)
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
        bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
        from src.utils import draw_bboxes
        img_out = draw_bboxes(img, bboxes, color=(0, 0, 255))
        st.image(img_out, caption='Detected Defects', channels='BGR')
        st.success('Detection complete!')
        st.write(f"**Number of detected defects:** {len(bboxes)}")
        if len(bboxes) == 0:
            st.warning('No defects detected. If this is unexpected, try another image.')
        if len(bboxes) > 0:
            st.write("**Bounding Boxes (x1, y1, x2, y2):**")
            data = []
            for i, bbox in enumerate(bboxes):
                st.write(f"{i+1}. {bbox}")
                data.append({'bbox': bbox})
            # Download CSV
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Detection Data (CSV)', csv, 'detections.csv', 'text/csv')
        # Download result image
        import io
        is_success, buffer = cv2.imencode(".jpg", img_out)
        if is_success:
            st.download_button('Download Result Image', buffer.tobytes(), 'result.jpg', 'image/jpeg')
    elif method == 'advanced classical':
        bboxes, closed = advanced_classical_detection(img)
        from src.utils import draw_bboxes
        img_out = draw_bboxes(img, bboxes, color=(255, 0, 0))
        st.image(img_out, caption='Detected Defects (Advanced Classical)', channels='BGR')
        st.image(closed, caption='Edge Map', channels='GRAY')
        st.success('Detection complete!')
        st.write(f"**Number of detected defects:** {len(bboxes)}")
        if len(bboxes) == 0:
            st.warning('No defects detected. If this is unexpected, try another image.')
        if len(bboxes) > 0:
            st.write("**Bounding Boxes (x1, y1, x2, y2):**")
            data = []
            for i, bbox in enumerate(bboxes):
                st.write(f"{i+1}. {bbox}")
                data.append({'bbox': bbox})
            # Download CSV
            df = pd.DataFrame(data)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Detection Data (CSV)', csv, 'detections.csv', 'text/csv')
        # Download result image
        import io
        is_success, buffer = cv2.imencode(".jpg", img_out)
        if is_success:
            st.download_button('Download Result Image', buffer.tobytes(), 'result.jpg', 'image/jpeg') 