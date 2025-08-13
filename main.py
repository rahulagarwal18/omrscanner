import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import urllib.request

# Enhanced constants for better accuracy
BUBBLE_THRESHOLD = 0.35  # Lowered threshold for better detection
MIN_BUBBLE_AREA = 100     # Minimum area for bubble detection
MAX_BUBBLE_AREA = 2500   # Maximum area for bubble detection
QUESTIONS_PER_ROW = 5
DEBUG_MODE = False
BLUR_KERNEL_SIZE = 3
MORPH_KERNEL_SIZE = 2

def order_points(pts):
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def enhance_and_correct_omr_image(image):
    """Enhanced image correction with better preprocessing"""
    original = image.copy()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Noise reduction
    gray = cv2.medianBlur(gray, 3)
    
    # Gentle blur for edge detection
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    
    # Adaptive threshold for better edge detection
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours for document detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 10000:
            doc_cnt = approx
            break
    
    if doc_cnt is None:
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
    
    if doc_cnt is None:
        raise Exception("Could not find document boundary. Please ensure the OMR sheet is clearly visible.")
    
    # Order points and perform perspective correction
    rect = order_points(doc_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    # Calculate dimensions
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    
    # Destination points for perspective transform
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    
    return warped

class OMRScanner:
    def _init_(self, answer_key):
        self.answer_key = answer_key
        self.total_questions = len(answer_key)
    
    def preprocess_image(self, image):
        """Enhanced preprocessing for better bubble detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # Use adaptive threshold for better detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def detect_bubbles(self, binary_image):
        """Detect bubble regions with improved accuracy"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_BUBBLE_AREA <= area <= MAX_BUBBLE_AREA:
                # Check if contour is roughly circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 <= circularity <= 1.5:  # More lenient circularity
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        if 0.5 <= aspect_ratio <= 2.0:  # More lenient aspect ratio
                            # Additional check: ensure minimum size
                            if w >= 12 and h >= 12:  # Minimum bubble size
                                bubbles.append((x, y, w, h))
        
        return bubbles
    
    def organize_bubbles(self, bubbles):
        """Organize bubbles by rows (questions) with improved logic"""
        if not bubbles:
            return {}
        
        # Sort bubbles by y-coordinate (top to bottom)
        bubbles.sort(key=lambda b: b[1])
        
        # Group bubbles by rows using clustering
        rows = {}
        current_row = 0
        
        if bubbles:
            # First bubble starts row 0
            rows[0] = [bubbles[0]]
            last_y = bubbles[0][1]
            
            for bubble in bubbles[1:]:
                x, y, w, h = bubble
                
                # If y-coordinate differs significantly, start new row
                if abs(y - last_y) > 25:  # Threshold for row detection
                    current_row += 1
                    rows[current_row] = []
                
                if current_row not in rows:
                    rows[current_row] = []
                    
                rows[current_row].append(bubble)
                last_y = y
        
        # Filter out rows with too few bubbles (noise)
        filtered_rows = {}
        for row_idx, row_bubbles in rows.items():
            if len(row_bubbles) >= 3:  # At least 3 bubbles per row
                # Sort each row by x-coordinate (left to right)
                row_bubbles.sort(key=lambda b: b[0])
                # Only take the expected number of bubbles per row
                filtered_rows[row_idx] = row_bubbles[:QUESTIONS_PER_ROW]
        
        return filtered_rows
    
    def is_bubble_filled(self, image, bubble):
        """Enhanced bubble fill detection with adjusted thresholds"""
        x, y, w, h = bubble
        
        # Extract bubble region with small padding
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        bubble_region = image[y_start:y_end, x_start:x_end]
        
        if bubble_region.size == 0 or bubble_region.shape[0] < 10 or bubble_region.shape[1] < 10:
            return False
        
        # Convert to grayscale if needed
        if len(bubble_region.shape) == 3:
            bubble_region_gray = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
        else:
            bubble_region_gray = bubble_region.copy()
        
        # Method 1: Mean intensity check - ADJUSTED THRESHOLD
        mean_intensity = np.mean(bubble_region_gray)
        intensity_threshold = 180  # Increased threshold (darker values indicate filled bubbles)
        
        # Method 2: Create circular mask and check fill percentage
        mask = np.zeros(bubble_region_gray.shape[:2], dtype=np.uint8)
        center = (bubble_region_gray.shape[1] // 2, bubble_region_gray.shape[0] // 2)
        radius = min(bubble_region_gray.shape[0], bubble_region_gray.shape[1]) // 3
        
        if radius < 4:
            return False
            
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply threshold to detect dark pixels - USING ADAPTIVE THRESHOLD
        thresh = cv2.adaptiveThreshold(bubble_region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply mask to thresholded image
        masked_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # Calculate fill percentage
        total_mask_pixels = cv2.countNonZero(mask)
        filled_pixels = cv2.countNonZero(masked_thresh)
        
        if total_mask_pixels == 0:
            return False
            
        fill_percentage = filled_pixels / total_mask_pixels
        
        # Method 3: Standard deviation check (filled bubbles have more uniform intensity)
        std_dev = np.std(bubble_region_gray[mask > 0])
        
        # Method 4: Edge density check (filled bubbles have fewer internal edges)
        edges = cv2.Canny(bubble_region_gray, 50, 150)
        edge_density = np.sum(edges[mask > 0]) / total_mask_pixels if total_mask_pixels > 0 else 0
        
        # ADJUSTED CRITERIA - More lenient
        intensity_filled = mean_intensity < intensity_threshold
        percentage_filled = fill_percentage > BUBBLE_THRESHOLD  # Using 0.35 now
        consistency_check = std_dev < 50  # Increased from 40
        edge_check = edge_density < 0.4  # Increased from 0.3
        
        # A bubble is considered filled if it meets at least 2 criteria (reduced from 3)
        criteria_met = sum([intensity_filled, percentage_filled, consistency_check, edge_check])
        is_filled = criteria_met >= 2  # Reduced from 3 to 2
        
        if DEBUG_MODE:
            print(f"Bubble at ({x}, {y}): mean_intensity={mean_intensity:.1f}, "
                  f"fill_percentage={fill_percentage:.3f}, std_dev={std_dev:.1f}, "
                  f"edge_density={edge_density:.3f}, criteria_met={criteria_met}, is_filled={is_filled}")
        
        return is_filled
    
    def scan_sheet_from_image(self, image):
        """Scan OMR sheet from image array"""
        if image is None or image.size == 0:
            raise Exception("Invalid image provided")
        
        # Preprocess the image
        binary = self.preprocess_image(image)
        
        # Detect bubbles
        bubbles = self.detect_bubbles(binary)
        
        if not bubbles:
            raise Exception("No bubbles detected. Please check image quality and bubble sizes.")
        
        # Organize bubbles by rows
        rows = self.organize_bubbles(bubbles)
        
        if not rows:
            raise Exception("Could not organize bubbles into rows.")
        
        # Analyze each row using original image
        detected_answers = {}
        student_answers = {}
        
        for row_idx, row_bubbles in rows.items():
            question_num = row_idx + 1
            if question_num > self.total_questions:
                break
            
            # Check which bubbles are filled
            filled_bubbles = []
            for bubble_idx, bubble in enumerate(row_bubbles):
                if self.is_bubble_filled(image, bubble):
                    filled_bubbles.append(bubble_idx)
            
            # Determine the answer
            if len(filled_bubbles) == 1:
                # Single bubble filled - normal case
                answer_idx = filled_bubbles[0]
                if answer_idx < 4:  # A, B, C, D
                    student_answers[question_num] = chr(ord('A') + answer_idx)
                else:
                    student_answers[question_num] = 'E'  # If 5 options
            elif len(filled_bubbles) == 0:
                # No bubble filled
                student_answers[question_num] = 'BLANK'
            else:
                # Multiple bubbles filled
                student_answers[question_num] = 'MULTIPLE'
        
        # Calculate score
        correct_answers = 0
        total_answers = len(self.answer_key)
        
        for question_num, correct_answer in self.answer_key.items():
            student_answer = student_answers.get(question_num, 'BLANK')
            if student_answer == correct_answer:
                correct_answers += 1
        
        percentage = (correct_answers / total_answers) * 100 if total_answers > 0 else 0
        
        results = {
            'student_answers': student_answers,
            'correct_answers': self.answer_key,
            'score': correct_answers,
            'total': total_answers,
            'percentage': percentage,
            'bubbles_detected': len(bubbles),
            'rows_detected': len(rows)
        }
        
        return results
    
    def scan_sheet(self, image_path):
        """Scan OMR sheet from file path"""
        if not os.path.exists(image_path):
            raise Exception(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Could not load image: {image_path}")
        
        return self.scan_sheet_from_image(image)
    
    def generate_report(self, results, student_name="Unknown", format_type="text"):
        """Generate detailed report"""
        if format_type == "json":
            return json.dumps({
                'student_name': student_name,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, indent=2)
        
        elif format_type == "csv":
            lines = ["Student Name,Question,Student Answer,Correct Answer,Result"]
            for q_num in sorted(results['correct_answers'].keys()):
                student_ans = results['student_answers'].get(q_num, 'BLANK')
                correct_ans = results['correct_answers'][q_num]
                result = "✓" if student_ans == correct_ans else "✗"
                lines.append(f"{student_name},{q_num},{student_ans},{correct_ans},{result}")
            return "\n".join(lines)
        
        else:  # text format
            report = []
            report.append("=" * 60)
            report.append(f"OMR SCAN REPORT")
            report.append("=" * 60)
            report.append(f"Student Name: {student_name}")
            report.append(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)")
            report.append(f"Bubbles Detected: {results['bubbles_detected']}")
            report.append(f"Rows Detected: {results['rows_detected']}")
            report.append("-" * 60)
            report.append("DETAILED RESULTS:")
            report.append("-" * 60)
            
            for q_num in sorted(results['correct_answers'].keys()):
                student_ans = results['student_answers'].get(q_num, 'BLANK')
                correct_ans = results['correct_answers'][q_num]
                status = "✓ CORRECT" if student_ans == correct_ans else "✗ INCORRECT"
                
                if student_ans == 'BLANK':
                    status = "⚠ BLANK"
                elif student_ans == 'MULTIPLE':
                    status = "⚠ MULTIPLE"
                
                report.append(f"Q{q_num:2d}: {student_ans:8s} (Correct: {correct_ans}) - {status}")
            
            report.append("=" * 60)
            return "\n".join(report)

def load_answer_key_from_file(filepath):
    """Load answer key from JSON or CSV file"""
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    
    elif filepath.endswith('.csv'):
        answer_key = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    question_num = int(parts[0])
                    answer = parts[1].strip().upper()
                    answer_key[question_num] = answer
        return answer_key
    
    else:
        raise Exception("Unsupported file format. Use JSON or CSV.")

def create_sample_answer_key():
    """Sample answer key for testing"""
    return {
        1: "A", 2: "C", 3: "B", 4: "D", 5: "E",
        6: "A", 7: "E", 8: "D", 9: "A", 10: "C"
    }

def run_gui():
    """Enhanced GUI with better error handling"""
    root = tk.Tk()
    root.title("OMR Scanner - Enhanced Accuracy")
    root.geometry("800x700")
    
    # Variables
    image_path = tk.StringVar()
    student_name = tk.StringVar(value="Unknown")
    ip_cam_url = tk.StringVar(value="http://192.168.1.100:8080/shot.jpg")
    answer_key = create_sample_answer_key()
    
    # Control Frame
    control_frame = tk.Frame(root, padx=10, pady=10)
    control_frame.pack(fill=tk.X)
    
    # Image selection
    tk.Label(control_frame, text="OMR Sheet:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5)
    tk.Entry(control_frame, textvariable=image_path, width=50).grid(row=0, column=1, padx=5)
    tk.Button(control_frame, text="Browse", command=lambda: image_path.set(
        filedialog.askopenfilename(title="Select OMR Sheet",
                                   filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp"), ("All files", ".*")])
    )).grid(row=0, column=2, padx=5)
    
    # Student name
    tk.Label(control_frame, text="Student Name:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=5)
    tk.Entry(control_frame, textvariable=student_name, width=50).grid(row=1, column=1, pady=5, padx=5)
    
    # IP Camera URL
    tk.Label(control_frame, text="IP Camera URL:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=5)
    tk.Entry(control_frame, textvariable=ip_cam_url, width=50).grid(row=2, column=1, pady=5, padx=5)
    
    # Debug mode checkbox
    debug_var = tk.BooleanVar()
    tk.Checkbutton(control_frame, text="Debug Mode", variable=debug_var).grid(row=3, column=0, sticky=tk.W, padx=5)
    
    # Results display
    results_text = scrolledtext.ScrolledText(root, height=25, width=90, font=("Courier", 9))
    results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def load_answer_key():
        nonlocal answer_key
        filename = filedialog.askopenfilename(
            title="Select Answer Key",
            filetypes=[("JSON files", ".json"), ("CSV files", ".csv"), ("All files", ".")]
        )
        if filename:
            try:
                answer_key = load_answer_key_from_file(filename)
                messagebox.showinfo("Success", f"Answer key loaded: {len(answer_key)} questions")
                results_text.insert(tk.END, f"Answer key loaded: {len(answer_key)} questions\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load answer key:\n{str(e)}")
    
    def scan_image():
        if not image_path.get():
            messagebox.showerror("Error", "Please select an OMR sheet image")
            return
        
        try:
            global DEBUG_MODE
            DEBUG_MODE = debug_var.get()
            
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, "Scanning image... Please wait.\n")
            root.update()
            
            scanner = OMRScanner(answer_key)
            results = scanner.scan_sheet(image_path.get())
            report = scanner.generate_report(results, student_name.get())
            
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, report)
            
            messagebox.showinfo("Scan Complete", 
                              f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                              f"Bubbles detected: {results['bubbles_detected']}\n"
                              f"Rows detected: {results['rows_detected']}")
            
        except Exception as e:
            messagebox.showerror("Scan Error", f"Error during scanning:\n{str(e)}")
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, f"Error: {str(e)}\n\nTroubleshooting tips:\n"
                               "1. Ensure the OMR sheet is well-lit and clear\n"
                               "2. Check that all four corners of the sheet are visible\n"
                               "3. Make sure bubbles are clearly marked\n"
                               "4. Try adjusting the image quality or resolution\n")
    
    def scan_from_camera():
        url = ip_cam_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter IP camera URL")
            return
        
        try:
            global DEBUG_MODE
            DEBUG_MODE = debug_var.get()
            
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, "Capturing from IP camera... Please wait.\n")
            root.update()
            
            # Capture image from IP camera
            resp = urllib.request.urlopen(url)
            image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if image is None:
                raise Exception("Could not decode image from IP camera")
            
            # Enhance and correct the image
            corrected_image = enhance_and_correct_omr_image(image)
            
            # Show corrected image
            cv2.imshow("Corrected OMR Sheet", corrected_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            
            # Scan the corrected image
            scanner = OMRScanner(answer_key)
            results = scanner.scan_sheet_from_image(corrected_image)
            report = scanner.generate_report(results, student_name.get())
            
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, report)
            
            messagebox.showinfo("Scan Complete", 
                              f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                              f"Bubbles detected: {results['bubbles_detected']}\n"
                              f"Rows detected: {results['rows_detected']}")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error scanning from camera:\n{str(e)}")
            results_text.delete(1.0, tk.END)
            results_text.insert(1.0, f"Camera Error: {str(e)}\n\nTroubleshooting tips:\n"
                               "1. Check IP camera URL format (e.g., http://192.168.1.100:8080/shot.jpg)\n"
                               "2. Ensure camera is accessible on the network\n"
                               "3. Make sure the OMR sheet is clearly visible in the camera view\n"
                               "4. Improve lighting conditions\n")
    
    def save_results():
        content = results_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showinfo("No Results", "No results to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", ".txt"), ("CSV files", ".csv"), ("JSON files", "*.json")]
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Results saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Load Answer Key", command=load_answer_key, 
              bg="#2196F3", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Scan Image", command=scan_image, 
              bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=20).pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Scan from Camera", command=scan_from_camera, 
              bg="#FF9800", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Save Results", command=save_results, 
              bg="#9C27B0", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)
    
    root.mainloop()

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Enhanced OMR Sheet Scanner")
    parser.add_argument("image", help="Path to OMR sheet image")
    parser.add_argument("--answer-key", help="Path to answer key JSON/CSV file")
    parser.add_argument("--student", default="Unknown", help="Student name")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # Load answer key
    if args.answer_key and os.path.exists(args.answer_key):
        answer_key = load_answer_key_from_file(args.answer_key)
        print(f"Loaded answer key: {len(answer_key)} questions")
    else:
        print("Using sample answer key...")
        answer_key = create_sample_answer_key()

    
    # Scan the image
        scanner = OMRScanner(answer_key)
    try:
        results = scanner.scan_sheet(args.image)
        report = scanner.generate_report(results, args.student, args.format)
        
        print(report)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "_main_":
    if len(sys.argv) == 1:
        print("OMR Scanner - Enhanced Accuracy Version")
        print("=====================================")
        print("Starting GUI Mode...")
        run_gui()
    else:
        exit(main())