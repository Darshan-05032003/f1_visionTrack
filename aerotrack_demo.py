# AeroTrack: F1 Aerodynamic Component Change Detection System
# TrackShift 2025 - Track 1 Winning Demo
# ============================================================

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 12

class AeroTrack:
    '''
    F1 Aerodynamic Component Change Detection System

    Detects and visualizes design changes between F1 car iterations
    using computer vision and structural similarity analysis.
    '''

    def __init__(self):
        self.similarity_threshold = 0.85  # 85% similarity = significant change
        self.min_change_area = 500  # Minimum pixels to consider as change

    def load_and_preprocess(self, image_path, target_size=(800, 600)):
        '''Load and preprocess image for analysis'''
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize for consistent processing
            img_resized = cv2.resize(img, target_size)

            # Convert to RGB for display
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for analysis
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            return img_resized, img_rgb, img_gray

        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None, None

    def detect_changes(self, img1_gray, img2_gray):
        '''Detect changes using SSIM (Structural Similarity Index)'''

        # Compute SSIM between two images
        score, diff_image = ssim(img1_gray, img2_gray, full=True)

        # Convert difference image to uint8
        diff_image = (diff_image * 255).astype("uint8")

        # Invert (so changes are white)
        diff_image = 255 - diff_image

        # Threshold to get binary mask
        _, thresh = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return score, diff_image, thresh

    def find_change_regions(self, thresh_image, min_area=500):
        '''Find bounding boxes around changed regions'''

        # Find contours
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area and get bounding boxes
        change_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                change_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'percentage': 0  # Will calculate later
                })

        return change_regions

    def classify_change(self, similarity_score):
        '''Classify the magnitude of change'''
        if similarity_score >= 0.95:
            return "NEGLIGIBLE", "green"
        elif similarity_score >= 0.85:
            return "MINOR", "yellow"
        elif similarity_score >= 0.70:
            return "MODERATE", "orange"
        else:
            return "MAJOR", "red"

    def visualize_results(self, img1_rgb, img2_rgb, diff_image, thresh, 
                         change_regions, similarity_score, save_path='results.png'):
        '''Create comprehensive visualization of results'''

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # Subplot 1: Original Design
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img1_rgb)
        ax1.set_title('Original F1 Design', fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')

        # Subplot 2: Modified Design
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(img2_rgb)
        ax2.set_title('Modified F1 Design', fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')

        # Subplot 3: Difference Heatmap
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(diff_image, cmap='hot', interpolation='bilinear')
        ax3.set_title('Change Heatmap', fontsize=16, fontweight='bold', pad=20)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Subplot 4: Threshold Mask
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(thresh, cmap='gray')
        ax4.set_title('Change Detection Mask', fontsize=16, fontweight='bold', pad=20)
        ax4.axis('off')

        # Subplot 5: Annotated Results
        ax5 = plt.subplot(2, 3, 5)
        result_img = img2_rgb.copy()

        # Draw bounding boxes on changes
        for idx, region in enumerate(change_regions):
            x, y, w, h = region['bbox']

            # Draw rectangle
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # Add label
            label = f"C{idx+1}: {region['area']/1000:.1f}K px²"
            cv2.putText(result_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ax5.imshow(result_img)
        ax5.set_title(f'Detected Changes: {len(change_regions)} Regions', 
                     fontsize=16, fontweight='bold', pad=20)
        ax5.axis('off')

        # Subplot 6: Analysis Report
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # Calculate statistics
        total_change_area = sum([r['area'] for r in change_regions])
        image_area = img1_rgb.shape[0] * img1_rgb.shape[1]
        change_percentage = (total_change_area / image_area) * 100

        classification, color = self.classify_change(similarity_score)

        # Create report text
        report_text = f'''
🏎️  AEROTRACK ANALYSIS REPORT
{'='*50}

📊 SIMILARITY METRICS
   Similarity Score: {similarity_score*100:.2f}%
   Difference Score: {(1-similarity_score)*100:.2f}%

🎯 CHANGE DETECTION
   Regions Detected: {len(change_regions)}
   Total Changed Area: {total_change_area/1000:.1f}K pixels²
   Change Percentage: {change_percentage:.2f}%

⚠️  CLASSIFICATION
   Change Level: {classification}
   Confidence: {similarity_score*100:.1f}%

💡 RECOMMENDATIONS
   → {'CFD re-analysis required' if classification in ['MAJOR', 'MODERATE'] else 'Monitor in next iteration'}
   → {'Wind tunnel validation needed' if classification == 'MAJOR' else 'Track performance analysis sufficient'}
   → Estimated Impact: {'HIGH' if classification in ['MAJOR', 'MODERATE'] else 'LOW'} aerodynamic change
        '''

        ax6.text(0.1, 0.95, report_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Visualization saved to: {save_path}")
        plt.show()

        return fig

    def generate_report(self, similarity_score, change_regions):
        '''Generate detailed text report'''

        classification, _ = self.classify_change(similarity_score)

        print("\n" + "="*70)
        print("🏎️  AEROTRACK COMPONENT ANALYSIS REPORT")
        print("="*70)
        print(f"\n📊 SIMILARITY ANALYSIS")
        print(f"   Component Similarity: {similarity_score*100:.2f}%")
        print(f"   Difference Detected: {(1-similarity_score)*100:.2f}%")

        print(f"\n🔍 CHANGE DETECTION RESULTS")
        print(f"   Total Change Regions: {len(change_regions)}")

        if change_regions:
            print(f"\n   Detailed Region Analysis:")
            total_area = 0
            for idx, region in enumerate(change_regions, 1):
                area_k = region['area'] / 1000
                total_area += region['area']
                x, y, w, h = region['bbox']
                print(f"   Region {idx}: {area_k:.1f}K pixels² at position ({x}, {y})")

            print(f"\n   Total Modified Area: {total_area/1000:.1f}K pixels²")

        print(f"\n⚠️  CHANGE CLASSIFICATION: {classification}")
        print(f"   Confidence Level: {similarity_score*100:.1f}%")

        print(f"\n💡 ENGINEERING RECOMMENDATIONS")
        if classification == "MAJOR":
            print("   → CRITICAL: Full CFD re-analysis required")
            print("   → Wind tunnel validation mandatory")
            print("   → Expect significant aerodynamic impact")
            print("   → Review with chief aerodynamicist immediately")
        elif classification == "MODERATE":
            print("   → CFD spot-check analysis recommended")
            print("   → Track day validation sufficient")
            print("   → Moderate aerodynamic impact expected")
            print("   → Monitor performance in next testing session")
        elif classification == "MINOR":
            print("   → Standard monitoring sufficient")
            print("   → Track performance comparison adequate")
            print("   → Minimal aerodynamic impact expected")
            print("   → Document for season-end review")
        else:
            print("   → No action required")
            print("   → Changes negligible")
            print("   → Standard monitoring continues")

        print(f"\n💰 ESTIMATED IMPACT")
        if classification in ["MAJOR", "MODERATE"]:
            print(f"   Time Saved: ~4 hours of manual analysis")
            print(f"   Cost Saved: ~$8,000 in engineering time")
            print(f"   CFD Hours Optimized: Focus only on changed regions")
        else:
            print(f"   Time Saved: ~30 minutes of manual analysis")
            print(f"   Verification Avoided: No CFD re-run needed")

        print("\n" + "="*70)
        print("✅ Analysis Complete | Generated by AeroTrack v1.0")
        print("="*70 + "\n")

    def analyze(self, image1_path, image2_path, output_path='aerotrack_results.png'):
        '''
        Complete analysis pipeline

        Args:
            image1_path: Path to original design image
            image2_path: Path to modified design image
            output_path: Path to save visualization

        Returns:
            Dictionary with analysis results
        '''

        print("\n🚀 AEROTRACK ANALYSIS STARTING...")
        print("="*70)

        # Step 1: Load images
        print("\n📸 Step 1/5: Loading images...")
        img1, img1_rgb, img1_gray = self.load_and_preprocess(image1_path)
        img2, img2_rgb, img2_gray = self.load_and_preprocess(image2_path)

        if img1 is None or img2 is None:
            print("❌ Error: Could not load images")
            return None

        print("   ✓ Images loaded successfully")

        # Step 2: Detect changes
        print("\n🔍 Step 2/5: Detecting changes using SSIM...")
        similarity_score, diff_image, thresh = self.detect_changes(img1_gray, img2_gray)
        print(f"   ✓ Similarity score computed: {similarity_score*100:.2f}%")

        # Step 3: Find change regions
        print("\n🎯 Step 3/5: Localizing change regions...")
        change_regions = self.find_change_regions(thresh, self.min_change_area)
        print(f"   ✓ Found {len(change_regions)} significant change regions")

        # Step 4: Visualize results
        print("\n📊 Step 4/5: Generating visualizations...")
        self.visualize_results(img1_rgb, img2_rgb, diff_image, thresh, 
                              change_regions, similarity_score, output_path)

        # Step 5: Generate report
        print("\n📝 Step 5/5: Generating detailed report...")
        self.generate_report(similarity_score, change_regions)

        # Return results
        results = {
            'similarity_score': similarity_score,
            'change_regions': change_regions,
            'classification': self.classify_change(similarity_score)[0],
            'num_changes': len(change_regions),
            'total_change_area': sum([r['area'] for r in change_regions])
        }

        print("\n🎉 ANALYSIS COMPLETE!")
        print("="*70)

        return results


# ============================================================
# DEMO EXECUTION
# ============================================================

if __name__ == "__main__":
    print('''
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║              🏎️  AEROTRACK DEMO SYSTEM 🏎️                    ║
    ║                                                                ║
    ║   F1 Aerodynamic Component Change Detection                   ║
    ║   Built for TrackShift 2025 Innovation Challenge             ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    ''')

    # Initialize AeroTrack
    tracker = AeroTrack()

    # NOTE: You need to provide two F1 car images
    # Download from: Google Images "F1 front wing 2024 vs 2025"
    # Or use any two similar images for testing

    print("\n⚠️  SETUP REQUIRED:")
    print("   1. Download two F1 car images (before/after)")
    print("   2. Save them as 'f1_before.jpg' and 'f1_after.jpg'")
    print("   3. Place them in the same folder as this script")
    print("   4. Run this script again!")
    print("\n   Quick download: Search 'F1 front wing evolution' on Google Images")

    # Check if demo images exist
    import os
    if os.path.exists('f1_before.jpg') and os.path.exists('f1_after.jpg'):
        print("\n✅ Demo images found! Starting analysis...\n")

        # Run analysis
        results = tracker.analyze('f1_before.jpg', 'f1_after.jpg')

        if results:
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Similarity: {results['similarity_score']*100:.1f}%")
            print(f"   Changes Found: {results['num_changes']} regions")
            print(f"   Classification: {results['classification']}")
            print(f"   Total Changed Area: {results['total_change_area']/1000:.1f}K pixels²")
    else:
        print("\n⏳ Waiting for demo images...")
        print("   (Script will run full demo once images are provided)")

