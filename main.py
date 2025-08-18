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

# Configuration constants
BUBBLE_THRESHOLD = 0.15  # Threshold that worked well in your tests
MIN_BUBBLE_AREA = 100
MAX_BUBBLE_AREA = 5000
QUESTIONS_PER_ROW = 5
DEBUG_MODE = False

# ---------------------------
# Geometry / preprocessing
# ---------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left.
    This is the standard mapping used for perspective transforms.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Standard convention
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def enhance_and_correct_omr_image(image: np.ndarray) -> np.ndarray:
    """Enhanced image correction with adaptive thresholding + perspective fix."""
    original = image.copy()

    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Contrast (CLAHE) + denoise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)

    # Smooth a bit for edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Morph cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find the largest 4-point contour (document)
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
        return original  # fall back

    rect = order_points(doc_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    return warped

# ---------------------------
# Core OMR Scanner
# ---------------------------

class OMRScanner:
    def __init__(self, answer_key: Dict[int, str]):
        """Initialize with an answer key like {1: 'A', 2: 'C', ...}."""
        # Normalize: int keys, upper-case values
        self.answer_key = {int(k): str(v).strip().upper() for k, v in answer_key.items()}
        self.debug_mode = DEBUG_MODE

    def load_answers(self, ans_file: str):
        with open(ans_file, "r") as f:
            data = json.load(f)
        self.answer_key = {int(k): str(v).strip().upper() for k, v in data.items()}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess to a clean binary suitable for contour detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize for consistency
        height, width = gray.shape
        if width > 800:
            scale = 800 / width
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu + Adaptive combined
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        combined = cv2.bitwise_or(otsu, adaptive)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        if self.debug_mode and (tk._default_root is None):  # avoid Tk conflicts
            cv2.imshow("Preprocessed", cleaned)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return cleaned

    def find_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect bubble contours and filter by dynamic area + circularity."""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles: List[Tuple[int, int, int, int]] = []
        areas: List[float] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                areas.append(area)

        if areas:
            areas.sort()
            median_area = areas[len(areas) // 2]
            min_area = max(median_area * 0.5, MIN_BUBBLE_AREA)
            max_area = min(median_area * 2.0, MAX_BUBBLE_AREA)
            print(f"Area statistics - Min: {min(areas)}, Max: {max(areas)}, Median: {median_area}")
            print(f"Using dynamic thresholds - Min: {min_area}, Max: {max_area}")
        else:
            min_area = MIN_BUBBLE_AREA
            max_area = MAX_BUBBLE_AREA

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / float(h)
                        if 0.6 < aspect_ratio < 1.4:
                            bubbles.append((x, y, w, h))

        print(f"Bubbles found after filtering: {len(bubbles)}")
        return bubbles

    def group_bubbles_by_question(self, bubbles: List[Tuple[int, int, int, int]]) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """Group bubbles into rows and map each row to a question (A–E)."""
        # Sort by Y asc, then X asc
        sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

        rows: List[List[Tuple[int, int, int, int]]] = []
        current_row: List[Tuple[int, int, int, int]] = []
        last_y = -1
        y_threshold = 30  # tolerance for same row

        for bubble in sorted_bubbles:
            if last_y == -1 or abs(bubble[1] - last_y) < y_threshold:  # use Y (index 1)
                current_row.append(bubble)
                last_y = bubble[1]
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))  # sort within row by X
                current_row = [bubble]
                last_y = bubble[1]

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))

        questions: Dict[int, List[Tuple[int, int, int, int]]] = {}
        question_num = 1

        for row in rows:
            if len(row) == QUESTIONS_PER_ROW:
                questions[question_num] = row
                question_num += 1
            elif len(row) > QUESTIONS_PER_ROW:
                for i in range(0, len(row), QUESTIONS_PER_ROW):
                    group = row[i:i + QUESTIONS_PER_ROW]
                    if len(group) == QUESTIONS_PER_ROW:
                        questions[question_num] = group
                        question_num += 1
        return questions

    def get_filled_answer(self, image: np.ndarray, bubbles: List[Tuple[int, int, int, int]]) -> Optional[str]:
        """Select which bubble (A–E) is most filled beyond threshold."""
        options = ['A', 'B', 'C', 'D', 'E']
        options = options[:min(len(options), len(bubbles))]

        max_fill_ratio = 0.0
        selected_answer: Optional[str] = None

        for idx, (x, y, w, h) in enumerate(bubbles):
            if idx >= len(options):
                break

            roi = image[y:y + h, x:x + w]
            mask = np.zeros(roi.shape, dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)

            masked = cv2.bitwise_and(roi, roi, mask=mask)
            total_pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(masked)

            if total_pixels > 0:
                fill_ratio = filled_pixels / total_pixels
                if fill_ratio > max_fill_ratio and fill_ratio > BUBBLE_THRESHOLD:
                    max_fill_ratio = fill_ratio
                    selected_answer = options[idx]
        return selected_answer

    def scan_sheet(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.scan_sheet_from_image(image)

    def scan_sheet_from_image(self, image: np.ndarray) -> Dict:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        # Try perspective/contrast correction first
        corrected = enhance_and_correct_omr_image(image)

        processed = self.preprocess_image(corrected)
        bubbles = self.find_bubbles(processed)

        if self.debug_mode and (tk._default_root is None):
            dbg = corrected.copy()
            for x, y, w, h in bubbles:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Detected Bubbles", dbg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"Total bubbles found: {len(bubbles)}")
        questions = self.group_bubbles_by_question(bubbles)
        print(f"Questions grouped: {len(questions)}")

        detected_answers: Dict[int, str] = {}
        for q_num, q_bubbles in questions.items():
            ans = self.get_filled_answer(processed, q_bubbles)
            if ans:
                detected_answers[q_num] = ans

        correct = 0
        total = len(self.answer_key)
        details = []

        for q_num in range(1, total + 1):
            detected = detected_answers.get(q_num, "N/A")
            correct_answer = self.answer_key.get(q_num, "N/A")
            is_correct = (detected == correct_answer)
            if is_correct:
                correct += 1
            details.append({
                "question": q_num,
                "detected": detected,
                "correct": correct_answer,
                "status": "✓" if is_correct else "✗" if detected != "N/A" else "-",
            })

        return {
            "score": correct,
            "total": total,
            "percentage": (correct / total * 100) if total > 0 else 0,
            "details": details,
            "detected_answers": detected_answers,
            "bubbles_detected": len(bubbles),
            "rows_detected": len(questions),
        }

    def generate_report(self, results: Dict, student_name: str = "Unknown", output_format: str = "text") -> str:
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
                "rows_detected": results.get("rows_detected", 0),
            }
            return json.dumps(report_data, indent=2)

        elif output_format == "csv":
            lines = ["Question,Detected,Correct,Status"]
            for d in results["details"]:
                lines.append(f"{d['question']},{d['detected']},{d['correct']},{d['status']}")
            lines.append(f"\nTotal Score,{results['score']}/{results['total']},{results['percentage']:.1f}%")
            return "\n".join(lines)

        # Default: Text report
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
        for d in results["details"]:
            report += f"Q{d['question']:2d}: {d['detected']:3s} (Correct: {d['correct']}) {d['status']}\n"
        return report

# ---------------------------
# Answer key IO
# ---------------------------

def create_sample_answer_key() -> Dict[int, str]:
    return {1: "B", 2: "D", 3: "A", 4: "C", 5: "B", 6: "A", 7: "C", 8: "D", 9: "B", 10: "A"}


def load_answer_key_from_file(filepath: str) -> Dict[int, str]:
    """Load answer key from JSON or CSV.
    JSON format: {"1": "A", "2": "B", ...}
    CSV format (flexible): question,answer OR question,<...>,answer
    """
    if filepath.lower().endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return {int(k): str(v).strip().upper() for k, v in data.items()}

    elif filepath.lower().endswith('.csv'):
        answer_key: Dict[int, str] = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # Try to skip header if it contains non-numeric first token
        start = 1 if lines and not lines[0].split(',')[0].strip().isdigit() else 0
        for line in lines[start:]:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 2 and parts[0].isdigit():
                qnum = int(parts[0])
                ans = parts[-1].upper()  # use last column for answer
                answer_key[qnum] = ans
        return answer_key

    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")

# ---------------------------
# GUI
# ---------------------------

def run_gui():
    try:
        root = tk.Tk()
        root.title("OMR Scanner - Enhanced with IP Camera & Multi-Scan")
        root.geometry("900x740")

        image_path = tk.StringVar()
        student_name = tk.StringVar(value="Unknown")
        ip_cam_url = tk.StringVar(value="http://10.150.114.196:8080/shot.jpg")
        answer_key = create_sample_answer_key()

        multi_files: List[str] = []

        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="OMR Sheet:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5)
        entry_path = tk.Entry(control_frame, textvariable=image_path, width=60)
        entry_path.grid(row=0, column=1, padx=5)

        def browse_file():
            filename = filedialog.askopenfilename(
                title="Select OMR Sheet",
                filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp"), ("All files", ".*")]
            )
            if filename:
                image_path.set(filename)

        tk.Button(control_frame, text="Browse", command=browse_file).grid(row=0, column=2, padx=5)

        def add_multiple_files():
            filenames = filedialog.askopenfilenames(
                title="Select Multiple OMR Sheets",
                filetypes=[("Image files", ".jpg *.jpeg *.png *.bmp"), ("All files", ".*")]
            )
            if filenames:
                multi_files.clear()
                multi_files.extend(list(filenames))
                messagebox.showinfo("Files Selected", f"Selected {len(multi_files)} files.")

        tk.Button(control_frame, text="Add Multiple Sheets", command=add_multiple_files).grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="Student Name:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=10)
        tk.Entry(control_frame, textvariable=student_name, width=60).grid(row=1, column=1, pady=10)

        tk.Label(control_frame, text="IP Camera URL:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=5)
        tk.Entry(control_frame, textvariable=ip_cam_url, width=60).grid(row=2, column=1, pady=5, padx=5)

        debug_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Debug Mode", variable=debug_var).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)

        tk.Label(root, text="Results:", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10)
        results_text = scrolledtext.ScrolledText(root, height=28, width=110, font=("Courier", 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def load_answer_key_gui():
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

        def scan_single():
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
                scanner.debug_mode = DEBUG_MODE
                results = scanner.scan_sheet(image_path.get())
                report = scanner.generate_report(results, student_name.get())

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, report)

                messagebox.showinfo(
                    "Scan Complete",
                    f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                    f"Bubbles detected: {results.get('bubbles_detected', 'N/A')}\n"
                    f"Rows detected: {results.get('rows_detected', 'N/A')}"
                )
            except Exception as e:
                messagebox.showerror("Error", str(e))
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, f"Error: {str(e)}\n")

        def scan_multiple_gui():
            if not multi_files:
                messagebox.showerror("Error", "Please add multiple OMR sheet images first")
                return
            try:
                global DEBUG_MODE
                DEBUG_MODE = debug_var.get()

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, "Scanning multiple sheets... Please wait.\n")
                root.update()

                scanner = OMRScanner(answer_key)
                scanner.debug_mode = DEBUG_MODE
                combined_report_parts: List[str] = []
                total_files = len(multi_files)

                for idx, fpath in enumerate(multi_files, start=1):
                    results_text.insert(tk.END, f"[{idx}/{total_files}] Scanning: {os.path.basename(fpath)}\n")
                    root.update()

                    results = scanner.scan_sheet(fpath)
                    per_file_report = scanner.generate_report(results, student_name.get(), "text")
                    header = f"\nFILE: {os.path.basename(fpath)}\n" + ("-" * 40) + "\n"
                    combined_report_parts.append(header + per_file_report)

                combined_report = "\n".join(combined_report_parts)
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, combined_report)

                messagebox.showinfo("Scan Complete", f"Scanned {total_files} files successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, f"Error: {str(e)}\n")

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

                resp = urllib.request.urlopen(url)
                image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is None:
                    raise Exception("Could not decode image from IP camera")

                corrected_image = enhance_and_correct_omr_image(image)

                if DEBUG_MODE and (tk._default_root is None):
                    cv2.imshow("Captured OMR Sheet", corrected_image)
                    cv2.waitKey(800)
                    cv2.destroyAllWindows()

                scanner = OMRScanner(answer_key)
                scanner.debug_mode = DEBUG_MODE
                results = scanner.scan_sheet_from_image(corrected_image)
                report = scanner.generate_report(results, student_name.get())

                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, report)

                messagebox.showinfo(
                    "Scan Complete",
                    f"Score: {results['score']}/{results['total']} ({results['percentage']:.1f}%)\n"
                    f"Bubbles detected: {results.get('bubbles_detected', 'N/A')}\n"
                    f"Rows detected: {results.get('rows_detected', 'N/A')}"
                )
            except Exception as e:
                messagebox.showerror("Camera Error", f"Error scanning from camera:\n{str(e)}")
                results_text.delete(1.0, tk.END)
                results_text.insert(1.0, f"Camera Error: {str(e)}\n")

        def save_results():
            content = results_text.get(1.0, tk.END).strip()
            if content:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", ".txt"), ("CSV files", ".csv"), ("JSON files", "*.json"), ("All files", ".")]
                )
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    messagebox.showinfo("Success", f"Results saved to {filename}")

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Load Answer Key", command=load_answer_key_gui,
                  bg="#2196F3", fg="white", font=("Arial", 10), padx=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Scan Image", command=scan_single,
                  bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=20).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Scan Multiple", command=scan_multiple_gui,
                  bg="#6A1B9A", fg="white", font=("Arial", 11, "bold"), padx=16).pack(side=tk.LEFT, padx=5)
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

# ---------------------------
# CLI entry
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="OMR Sheet Scanner with IP Camera & Multi-Scan")
    parser.add_argument("image", nargs="?", help="Path to OMR sheet image")
    parser.add_argument("--images", nargs="+", help="Paths to multiple OMR sheet images")
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

    scanner = OMRScanner(answer_key)
    scanner.debug_mode = DEBUG_MODE

    try:
        # Batch mode
        if args.images:
            combined_outputs: List[str] = []
            for fpath in args.images:
                if not os.path.exists(fpath):
                    print(f"Warning: File not found, skipping: {fpath}")
                    continue
                results = scanner.scan_sheet(fpath)
                header = f"\nFILE: {os.path.basename(fpath)}\n" + ("-" * 40) + "\n"
                per_file = scanner.generate_report(results, args.student, args.format)
                combined_outputs.append(header + per_file)

            report = "\n".join(combined_outputs) if combined_outputs else "No valid files processed."
            print(report)
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\nReport saved to: {args.output}")
            return 0

        # Single-image CLI or GUI fallback
        if not args.image:
            print("OMR Scanner - Starting GUI Mode")
            print("===============================")
            run_gui()
            return 0

        results = scanner.scan_sheet(args.image)
        report = scanner.generate_report(results, args.student, args.format)
        print(report)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("OMR Scanner - Starting GUI Mode")
        print("===============================")
        run_gui()
    else:
        sys.exit(main())