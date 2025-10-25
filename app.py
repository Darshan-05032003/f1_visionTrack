from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VisionTrack:
    def __init__(self):
        self.similarity_threshold = 0.85
        self.min_change_area = 500

    def load_and_preprocess(self, image_path, target_size=(800, 600)):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None, None

            img_resized = cv2.resize(img, target_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            return img_resized, img_rgb, img_gray
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def detect_changes(self, img1_gray, img2_gray):
        score, diff_image = ssim(img1_gray, img2_gray, full=True)
        diff_image = (diff_image * 255).astype("uint8")
        diff_image = 255 - diff_image

        _, thresh = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return score, diff_image, thresh

    def find_change_regions(self, thresh_image):
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        change_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_change_area:
                x, y, w, h = cv2.boundingRect(contour)
                change_regions.append({
                    'bbox': (x, y, w, h),
                    'area': int(area)
                })

        return sorted(change_regions, key=lambda x: x['area'], reverse=True)

    def classify_change(self, similarity_score):
        if similarity_score >= 0.95:
            return "NEGLIGIBLE"
        elif similarity_score >= 0.85:
            return "MINOR"
        elif similarity_score >= 0.70:
            return "MODERATE"
        else:
            return "MAJOR"

    def create_visualization(self, img2_rgb, change_regions, output_path):
        result_img = img2_rgb.copy()

        for idx, region in enumerate(change_regions[:5]):
            x, y, w, h = region['bbox']
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"C{idx+1}: {region['area']/1000:.1f}K"
            cv2.putText(result_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(result_rgb).save(output_path)

        return output_path

    def analyze(self, image1_path, image2_path):
        img1, img1_rgb, img1_gray = self.load_and_preprocess(image1_path)
        img2, img2_rgb, img2_gray = self.load_and_preprocess(image2_path)

        if img1 is None or img2 is None:
            return None

        similarity_score, diff_image, thresh = self.detect_changes(img1_gray, img2_gray)
        change_regions = self.find_change_regions(thresh)

        result_filename = f"result_{os.path.basename(image1_path)}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        self.create_visualization(img2_rgb, change_regions, result_path)

        total_area = sum([r['area'] for r in change_regions])
        classification = self.classify_change(similarity_score)

        results = {
            'similarity': round(similarity_score * 100, 2),
            'difference': round((1 - similarity_score) * 100, 2),
            'num_changes': len(change_regions),
            'total_area': round(total_area / 1000, 1),
            'classification': classification,
            'regions': [
                {
                    'id': idx + 1,
                    'area': round(r['area'] / 1000, 1),
                    'position': f"({r['bbox'][0]}, {r['bbox'][1]})"
                }
                for idx, r in enumerate(change_regions[:5])
            ],
            'result_image': result_filename
        }

        return results

tracker = VisionTrack()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No files selected'}), 400

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file format'}), 400

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    file1.save(filepath1)
    file2.save(filepath2)

    results = tracker.analyze(filepath1, filepath2)

    if results is None:
        return jsonify({'error': 'Failed to analyze images'}), 500

    return jsonify(results)

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("VISIONTRACK SERVER STARTING...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
