#!/usr/bin/env python3
"""
OMR Sheet Scanner - Merged Version
Combines the accurate detection model from original with enhanced features
"""

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

# Configuration constants (from original for accuracy)
BUBBLE_THRESHOLD = 0.15  # Original threshold that worked perfectly
MIN_BUBBLE_AREA = 100
MAX_BUBBLE_AREA = 5000  # Original max area
QUESTIONS_PER_ROW = 5
DEBUG_MODE = False


def order_points(pts):
    """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Noise reduction
    gray = cv2.medianBlur(gray, 3)

    # Gentle blur for edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold for better edge detection
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
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
        # If no document boundary found, return original image
        return original

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
    def __init__(self, answer_key: Dict[int, str]):
        self.answer_key = answer_key
        self.debug_mode = False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Original preprocessing method that worked perfectly"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize for consistent processing
        height, width = gray.shape
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try multiple thresholding methods (original approach)
        # Method 1: Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Method 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Combine both methods
        combined = cv2.bitwise_or(otsu, adaptive)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        if DEBUG_MODE:
            cv2.imshow("Preprocessed", cleaned)
            cv2.waitKey(0)

        return cleaned

    def find_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Original bubble detection method with dynamic thresholds"""
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bubbles = []
        areas = []

        # First pass: collect all potential bubbles
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Very low threshold initially
                areas.append(area)

        # Calculate dynamic thresholds based on detected areas
        if areas:
            areas.sort()
            # Use median area as reference
            median_area = areas[len(areas) // 2]
            min_area = median_area * 0.5
            max_area = median_area * 2.0

            print(f"Area statistics - Min: {min(areas)}, Max: {max(areas)}, Median: {median_area}")
            print(f"Using dynamic thresholds - Min: {min_area}, Max: {max_area}")
        else:
            min_area = MIN_BUBBLE_AREA
            max_area = MAX_BUBBLE_AREA

        # Second pass: filter bubbles
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if min_area < area < max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # More lenient circularity check
                    if circularity > 0.5:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / float(h)

                        # Check aspect ratio
                        if 0.6 < aspect_ratio < 1.4:
                            bubbles.append((x, y, w, h))

        print(f"Bubbles found after filtering: {len(bubbles)}")
        return bubbles

    def group_bubbles_by_question(self, bubbles: List[Tuple[int, int, int, int]]) -> Dict[
        int, List[Tuple[int, int, int, int]]]:
        """Original grouping method that worked perfectly"""
        # Sort bubbles by Y coordinate (top to bottom), then X (left to right)
        sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

        # Group bubbles by row
        rows = []
        current_row = []
        last_y = -1
        y_threshold = 30  # Maximum Y difference to be considered same row

        for bubble in sorted_bubbles:
            if last_y == -1 or abs(bubble[1] - last_y) < y_threshold:
                current_row.append(bubble)
                last_y = bubble[1]
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [bubble]
                last_y = bubble[1]

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        # Assign question numbers
        questions = {}
        question_num = 1

        for row in rows:
            # Each row should have QUESTIONS_PER_ROW bubbles
            if len(row) == QUESTIONS_PER_ROW:
                questions[question_num] = row
                question_num += 1
            elif len(row) > QUESTIONS_PER_ROW:
                # Handle rows with extra bubbles (possibly noise)
                # Group by proximity
                for i in range(0, len(row), QUESTIONS_PER_ROW):
                    group = row[i:i + QUESTIONS_PER_ROW]
                    if len(group) == QUESTIONS_PER_ROW:
                        questions[question_num] = group
                        question_num += 1

        return questions

    def get_filled_answer(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Optional[str]:
        """Original answer detection method"""
        options = ['A', 'B', 'C', 'D', 'E']
        max_fill_ratio = 0
        selected_answer = None

        for idx, (x, y, w, h) in enumerate(bubbles):
            if idx >= len(options):
                break

            # Extract bubble region
            bubble_roi = image[y:y + h, x:x + w]

            # Create circular mask
            mask = np.zeros(bubble_roi.shape, dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)

            # Calculate fill ratio
            masked_bubble = cv2.bitwise_and(bubble_roi, bubble_roi, mask=mask)
            total_pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(masked_bubble)

            if total_pixels > 0:
                fill_ratio = filled_pixels / total_pixels

                if fill_ratio > max_fill_ratio and fill_ratio > BUBBLE_THRESHOLD:
                    max_fill_ratio = fill_ratio
                    selected_answer = options[idx]

        return selected_answer

    def scan_sheet(self, image_path: str) -> Dict:
        """Main scanning function from file path"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.scan_sheet_from_image(image)

    def scan_sheet_from_image(self, image: np.ndarray) -> Dict:
        """Scan from image array (for IP camera support)"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        # Keep original for processing
        original = image.copy()

        # Preprocess
        processed = self.preprocess_image(image)

        # Find bubbles
        bubbles = self.find_bubbles(processed)

        if DEBUG_MODE:
            debug_img = original.copy()
            for x, y, w, h in bubbles:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Detected Bubbles", debug_img)
            cv2.waitKey(0)

        print(f"Total bubbles found: {len(bubbles)}")

        # Group by question
        questions = self.group_bubbles_by_question(bubbles)
        print(f"Questions grouped: {len(questions)}")

        # Extract answers
        detected_answers = {}
        for q_num, q_bubbles in questions.items():
            answer = self.get_filled_answer(processed, q_bubbles)
            if answer:
                detected_answers[q_num] = answer

        # Calculate score
        correct = 0
        total = len(self.answer_key)
        results = []

        for q_num in range(1, total + 1):
            detected = detected_answers.get(q_num, "N/A")
            correct_answer = self.answer_key.get(q_num, "N/A")
            is_correct = detected == correct_answer

            if is_correct:
                correct += 1

            results.append({
                "question": q_num,
                "detected": detected,
                "correct": correct_answer,
                "status": "✓" if is_correct else "✗" if detected != "N/A" else "-"
            })

        return {
            "score": correct,
            "total": total,
            "percentage": (correct / total * 100) if total > 0 else 0,
            "details": results,
            "detected_answers": detected_answers,
            "bubbles_detected": len(bubbles),
            "rows_detected": len(questions)
        }

    def generate_report(self, results: Dict, student_name: str = "Unknown", output_format: str = "text") -> str:
        """Generate a report in various formats"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if output_format == "json":
            report_data = {
                "student_name": student_name,
                "timestamp": timestamp,
                "score": results["score"],
                "total": results["total"],
                "percentage": results["percentage"],
                "answers": results["detected_answers"],
                "bubbles_detected": results.get("bubbles_detected", 0),
                "rows_detected": results.get("rows_detected", 0)
            }
            return json.dumps(report_data, indent=2)

        elif output_format == "csv":
            lines = ["Question,Detected,Correct,Status"]
            for detail in results["details"]:
                lines.append(f"{detail['question']},{detail['detected']},{detail['correct']},{detail['status']}")
            lines.append(f"\nTotal Score,{results['score']}/{results['total']},{results['percentage']:.1f}%")
            return "\n".join(lines)

        else:  # Default text format
            report = f"""
OMR SCAN REPORT
===============
Student: {student_name}
Date: {timestamp}
Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)
Bubbles Detected: {results.get('bubbles_detected', 'N/A')}
Rows Detected: {results.get('rows_detected', 'N/A')}

DETAILED RESULTS:
-----------------
"""
            for detail in results["details"]:
                report += f"Q{detail['question']:2d}: {detail['detected']:3s} (Correct: {detail['correct']}) {detail['status']}\n"

            return report


def create_sample_answer_key() -> Dict[int, str]:
    """Create a sample answer key"""
    return {
        1: "B", 2: "D", 3: "A", 4: "C", 5: "B",
        6: "A", 7: "C", 8: "D", 9: "B", 10: "A"
    }


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


def run_gui():
    """Enhanced GUI with IP camera support"""
    try:
        root = tk.Tk()
        root.title("OMR Scanner - Enhanced with IP Camera Support")
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
        entry_path = tk.Entry(control_frame, textvariable=image_path, width=50)
        entry_path.grid(row=0, column=1, padx=5)

        def browse_file():
            filename = filedialog.askopenfilename(
                title="Select OMR Sheet",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
            )
            if filename:
                image_path.set(filename)

        tk.Button(control_frame, text="Browse", command=browse_file).grid(row=0, column=2)

        # Student name
        tk.Label(control_frame, text="Student Name:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=5,
                                                                               pady=10)
        tk.Entry(control_frame, textvariable=student_name, width=50).grid(row=1, column=1, pady=10)

        # IP Camera URL
        tk.Label(control_frame, text="IP Camera URL:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=5)
        tk.Entry(control_frame, textvariable=ip_cam_url, width=50).grid(row=2, column=1, pady=5, padx=5)

        # Debug mode checkbox
        debug_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Debug Mode", variable=debug_var).grid(row=3, column=0, sticky=tk.W, padx=5,
                                                                                  pady=5)

        # Results area
        tk.Label(root, text="Results:", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10)
        results_text = scrolledtext.ScrolledText(root, height=25, width=90, font=("Courier", 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def load_answer_key():
            nonlocal answer_key
            filename = filedialog.askopenfilename(
                title="Select Answer Key",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                try:
                    answer_key = load_answer_key_from_file(filename)
                    messagebox.showinfo("Success", f"Answer key loaded: {len(answer_key)} questions")
                    results_text.insert(tk.END, f"Answer key loaded: {len(answer_key)} questions\n")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load answer key:\n{str(e)}")

        def scan():
            if not image_path.get():
                messagebox.showerror("Error", "Please select an OMR sheet image")
                return

            try:
                global DEBUG_MODE
                DEBUG_MODE = debug_var.get()

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, "Scanning... Please wait.\n")
                root.update()

                scanner = OMRScanner(answer_key)
                results = scanner.scan_sheet(image_path.get())
                report = scanner.generate_report(results, student_name.get())

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, report)

                # Show score in message box
                messagebox.showinfo("Scan Complete",
                                    f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                                    f"Bubbles detected: {results.get('bubbles_detected', 'N/A')}\n"
                                    f"Rows detected: {results.get('rows_detected', 'N/A')}")

            except Exception as e:
                messagebox.showerror("Error", str(e))
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, f"Error: {str(e)}\n\nTroubleshooting tips:\n"
                                         "1. Ensure the OMR sheet is well-lit and clear\n"
                                         "2. Check that all bubbles are visible\n"
                                         "3. Try enabling Debug Mode to see intermediate steps")

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

                # Try to enhance and correct the image
                try:
                    corrected_image = enhance_and_correct_omr_image(image)
                except:
                    # If correction fails, use original image
                    corrected_image = image

                # Show corrected image briefly
                if DEBUG_MODE:
                    cv2.imshow("Captured OMR Sheet", corrected_image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

                # Scan the image
                scanner = OMRScanner(answer_key)
                results = scanner.scan_sheet_from_image(corrected_image)
                report = scanner.generate_report(results, student_name.get())

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, report)

                messagebox.showinfo("Scan Complete",
                                    f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                                    f"Bubbles detected: {results.get('bubbles_detected', 'N/A')}\n"
                                    f"Rows detected: {results.get('rows_detected', 'N/A')}")

            except Exception as e:
                messagebox.showerror("Camera Error", f"Error scanning from camera:\n{str(e)}")
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, f"Camera Error: {str(e)}\n\nTroubleshooting tips:\n"
                                         "1. Check IP camera URL format (e.g., http://192.168.1.100:8080/shot.jpg)\n"
                                         "2. Ensure camera is accessible on the network\n"
                                         "3. Make sure the OMR sheet is clearly visible in the camera view\n"
                                         "4. Improve lighting conditions")

        # Save button
        def save_results():
            if results_text.get(1.0, tk.END).strip():
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("JSON files", "*.json"),
                               ("All files", "*.*")]
                )
                if filename:
                    with open(filename, 'w') as f:
                        f.write(results_text.get(1.0, tk.END))
                    messagebox.showinfo("Success", f"Results saved to {filename}")

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Load Answer Key", command=load_answer_key,
                  bg="#2196F3", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Scan Image", command=scan,
                  bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=20).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Scan from Camera", command=scan_from_camera,
                  bg="#FF9800", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Save Results", command=save_results,
                  bg="#9C27B0", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)

        root.mainloop()

    except ImportError:
        print("Tkinter not available. Please install it or use command line mode.")
        print("On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("On Windows: Tkinter should be included with Python")
        return


def main():
    parser = argparse.ArgumentParser(description="OMR Sheet Scanner with IP Camera Support")
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

    # Create scanner
    scanner = OMRScanner(answer_key)

    try:
        # Scan sheet
        results = scanner.scan_sheet(args.image)

        # Generate report
        report = scanner.generate_report(results, args.student, args.format)

        # Output results
        print(report)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # If no command line arguments, start GUI
    if len(sys.argv) == 1:
        print("OMR Scanner - Starting GUI Mode")
        print("===============================")
        run_gui()
    else:
        # Command line mode
        exit(main())