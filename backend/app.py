# type: ignore
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import urllib.request
from werkzeug.utils import secure_filename
import tempfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from docx import Document
from docx.shared import Inches
import io
import zipfile
import sqlite3
import sys
import traceback
from werkzeug.middleware.shared_data import SharedDataMiddleware

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Configuration constants - ADJUSTED FOR MULTIPLE DETECTION
BUBBLE_THRESHOLD = 0.20  # Increased slightly to reduce false positives
MIN_BUBBLE_AREA = 100
MAX_BUBBLE_AREA = 5000
QUESTIONS_PER_ROW = 5
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
DATABASE_FILE = 'omr_results.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
DEBUG_MODE = False

# ADDED: Dynamic threshold calculation settings
DYNAMIC_THRESHOLD = True  # Enable dynamic threshold calculation
RELATIVE_THRESHOLD = 0.7  # Bubble must be at least 70% as filled as the most filled bubble

# Create directories with proper permissions
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        try:
            os.chmod(folder, 0o755)
        except:
            pass

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Serve static files properly
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/downloads': app.config['RESULTS_FOLDER']
})

# FRONTEND SERVING ROUTES - Added to serve React app
@app.route('/')
def serve_frontend():
    """Serve the React frontend"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
    if os.path.exists(os.path.join(frontend_path, 'index.html')):
        return send_from_directory(frontend_path, 'index.html')
    else:
        # Fallback to API response if frontend not built
        return jsonify({
            "message": "Enhanced OMR Scanner API - Backend is running",
            "status": "Frontend not found. Please ensure frontend is built in frontend/dist.",
            "api_docs": "/api/health for API status"
        })

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from React build"""
    # Don't serve API routes as static files
    if path.startswith('api/'):
        # Let Flask handle API routes normally - this will fall through to API handlers
        pass
    else:
        frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
        
        # Try to serve the requested file
        file_path = os.path.join(frontend_path, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return send_from_directory(frontend_path, path)
        
        # For React Router - return index.html for client-side routing
        index_path = os.path.join(frontend_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(frontend_path, 'index.html')
        else:
            return jsonify({"error": "Frontend not built"}), 404

# Database setup for persistent storage
def init_database():
    """Initialize SQLite database for storing results"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            reg_no TEXT,
            class TEXT,
            timestamp TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            detected_answers TEXT,
            details TEXT,
            bubbles_detected INTEGER,
            rows_detected INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS answer_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            answer_key TEXT NOT NULL,
            created_at TEXT,
            is_current INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def calculate_answer_statistics(details):
    """Calculate correct, wrong, blank, and multiple answer counts"""
    correct_answers = 0
    wrong_answers = 0
    blank_answers = 0
    multiple_answers = 0
    
    for detail in details:
        status = detail.get('status', '-')
        detected = detail.get('detected', 'N/A')
        
        if status == '✓':
            correct_answers += 1
        elif detected == 'MULTIPLE' or status == 'M':
            multiple_answers += 1
        elif detected == 'N/A' or detected == 'BLANK' or status == '-':
            blank_answers += 1
        elif status == '✗':
            wrong_answers += 1
    
    return correct_answers, wrong_answers, blank_answers, multiple_answers

def save_student_result(student_data):
    """Save student result to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    results = student_data['results']
    cursor.execute('''
        INSERT INTO students (name, reg_no, class, timestamp, score, total, percentage, 
                            detected_answers, details, bubbles_detected, rows_detected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        student_data['name'],
        student_data['reg_no'],
        student_data['class'],
        student_data['timestamp'],
        results['score'],
        results['total'],
        results['percentage'],
        json.dumps(results['detected_answers']),
        json.dumps(results['details']),
        results.get('bubbles_detected', 0),
        results.get('rows_detected', 0)
    ))
    
    conn.commit()
    student_id = cursor.lastrowid
    conn.close()
    return student_id

def get_all_student_results():
    """Get all student results from database with flattened structure"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        details = json.loads(row[9])  # details field
        
        # Calculate answer statistics
        correct_answers, wrong_answers, blank_answers, multiple_answers = calculate_answer_statistics(details)
        
        result = {
            'id': row[0],
            'student_name': row[1],
            'reg_no': row[2],
            'class': row[3],
            'timestamp': row[4],
            'score': row[5],
            'total_questions': row[6],
            'total': row[6],
            'percentage': row[7],
            'detected_answers': json.loads(row[8]),
            'question_details': details,
            'details': details,
            'bubbles_detected': row[10],
            'rows_detected': row[11],
            'correct_answers': correct_answers,
            'wrong_answers': wrong_answers,
            'blank_answers': blank_answers,
            'multiple_answers': multiple_answers
        }
        results.append(result)
    
    conn.close()
    return results

def delete_student_result(result_id):
    """Delete a single student result"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM students WHERE id = ?', (result_id,))
    affected_rows = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return affected_rows > 0

def clear_all_student_results():
    """Clear all student results"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM students')
    affected_rows = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return affected_rows

def save_answer_key(name, answer_key, set_as_current=True):
    """Save answer key to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    if set_as_current:
        cursor.execute('UPDATE answer_keys SET is_current = 0')
    
    cursor.execute('''
        INSERT INTO answer_keys (name, answer_key, created_at, is_current)
        VALUES (?, ?, ?, ?)
    ''', (name, json.dumps(answer_key), datetime.now().isoformat(), 1 if set_as_current else 0))
    
    conn.commit()
    conn.close()

def get_current_answer_key():
    """Get current answer key from database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT answer_key FROM answer_keys WHERE is_current = 1 ORDER BY created_at DESC LIMIT 1')
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return json.loads(row[0])
    return None

# Initialize database
init_database()

# OMR Scanner classes with FIXED multiple detection
def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def enhance_and_correct_omr_image(image: np.ndarray) -> np.ndarray:
    """Enhanced image correction with adaptive thresholding + perspective fix."""
    original = image.copy()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, 3)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

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
        return original

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

class OMRScanner:
    def __init__(self, answer_key: Dict[int, str]):
        self.answer_key = {int(k): str(v).strip().upper() for k, v in answer_key.items()}
        self.debug_mode = DEBUG_MODE

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        if width > 800:
            scale = 800 / width
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        combined = cv2.bitwise_or(otsu, adaptive)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def find_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
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

        return bubbles

    def group_bubbles_by_question(self, bubbles: List[Tuple[int, int, int, int]]) -> Dict[int, List[Tuple[int, int, int, int]]]:
        sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
        rows: List[List[Tuple[int, int, int, int]]] = []
        current_row: List[Tuple[int, int, int, int]] = []
        last_y = -1
        y_threshold = 30

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
        """FIXED: Properly detect multiple filled bubbles"""
        options = ['A', 'B', 'C', 'D', 'E']
        options = options[:min(len(options), len(bubbles))]

        filled_answers = []
        fill_ratios = []

        # Calculate fill ratio for each bubble
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
                fill_ratios.append(fill_ratio)
                
                if self.debug_mode:
                    print(f"  Option {options[idx]}: Fill ratio = {fill_ratio:.3f}")
            else:
                fill_ratios.append(0.0)

        # Determine which bubbles are filled
        if DYNAMIC_THRESHOLD and fill_ratios:
            # Dynamic threshold: Consider a bubble filled if it's at least X% as filled as the most filled bubble
            max_fill = max(fill_ratios)
            threshold = max(BUBBLE_THRESHOLD, max_fill * RELATIVE_THRESHOLD)
            
            for idx, ratio in enumerate(fill_ratios):
                if idx < len(options) and ratio >= threshold:
                    filled_answers.append(options[idx])
                    
            if self.debug_mode:
                print(f"  Dynamic threshold: {threshold:.3f} (max fill: {max_fill:.3f})")
        else:
            # Static threshold
            for idx, ratio in enumerate(fill_ratios):
                if idx < len(options) and ratio > BUBBLE_THRESHOLD:
                    filled_answers.append(options[idx])

        # Return result based on number of filled bubbles
        if len(filled_answers) == 0:
            return None  # No answer (blank)
        elif len(filled_answers) == 1:
            return filled_answers[0]  # Single answer
        else:
            if self.debug_mode:
                print(f"  Multiple answers detected: {filled_answers}")
            return "MULTIPLE"  # Multiple answers detected

    def scan_answer_key_from_image(self, image: np.ndarray) -> Dict[int, str]:
        processed = self.preprocess_image(image)
        bubbles = self.find_bubbles(processed)
        questions = self.group_bubbles_by_question(bubbles)

        answer_key = {}
        for q_num, q_bubbles in questions.items():
            ans = self.get_filled_answer(processed, q_bubbles)
            if ans and ans != "MULTIPLE":  # Don't accept multiple answers in answer key
                answer_key[q_num] = ans

        return answer_key

    def scan_sheet_from_image(self, image: np.ndarray) -> Dict:
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        corrected = enhance_and_correct_omr_image(image)
        processed = self.preprocess_image(corrected)
        bubbles = self.find_bubbles(processed)
        questions = self.group_bubbles_by_question(bubbles)

        detected_answers: Dict[int, str] = {}
        
        if self.debug_mode:
            print(f"Scanning sheet with {len(questions)} questions detected")
        
        for q_num, q_bubbles in questions.items():
            if self.debug_mode:
                print(f"Question {q_num}:")
            ans = self.get_filled_answer(processed, q_bubbles)
            if ans:
                detected_answers[q_num] = ans

        correct = 0
        total = len(self.answer_key)
        details = []

        for q_num in range(1, total + 1):
            detected = detected_answers.get(q_num, "N/A")
            correct_answer = self.answer_key.get(q_num, "N/A")
            
            # Only count as correct if single answer matches
            is_correct = (detected == correct_answer) and detected != "MULTIPLE"
            
            if is_correct:
                correct += 1
                
            # Determine status symbol
            if detected == "MULTIPLE":
                status = "M"  # Multiple marked
            elif detected == "N/A" or detected == "BLANK":
                status = "-"  # Blank
            elif is_correct:
                status = "✓"  # Correct
            else:
                status = "✗"  # Wrong
                
            details.append({
                "question": q_num,
                "detected": detected,
                "correct": correct_answer,
                "status": status,
                "is_correct": is_correct,
                "detected_answer": detected,
                "correct_answer": correct_answer
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

# Export Functions (keeping existing implementations)
def export_to_pdf(student_data: Dict) -> str:
    """Export student results to PDF"""
    try:
        safe_name = "".join(c for c in str(student_data.get('student_name', student_data.get('name', 'Unknown'))) if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"Student_{student_data.get('id', datetime.now().strftime('%H%M%S'))}"
        
        filename = f"omr_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        c = canvas.Canvas(filepath, pagesize=A4)
        width, height = A4
        
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(width/2, height-60, "OMR SCAN REPORT")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height-120, "Student Information:")
        
        c.setFont("Helvetica", 12)
        y_pos = height - 140
        
        info_lines = [
            f"Name: {student_data.get('student_name', student_data.get('name', 'N/A'))}",
            f"Registration No: {student_data.get('reg_no', 'N/A')}",
            f"Class: {student_data.get('class', 'N/A')}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Score: {student_data.get('score', 0)}/{student_data.get('total_questions', student_data.get('total', 0))} ({student_data.get('percentage', 0):.1f}%)",
            f"Multiple Marked: {student_data.get('multiple_answers', 0)}"
        ]
        
        for line in info_lines:
            c.drawString(72, y_pos, line)
            y_pos -= 20
        
        y_pos -= 20
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y_pos, "Question Details:")
        
        y_pos -= 30
        c.setFont("Helvetica-Bold", 10)
        c.drawString(72, y_pos, "Question")
        c.drawString(150, y_pos, "Detected")
        c.drawString(220, y_pos, "Correct")
        c.drawString(290, y_pos, "Status")
        
        y_pos -= 5
        c.line(72, y_pos, 350, y_pos)
        
        c.setFont("Helvetica", 10)
        y_pos -= 15
        
        details = student_data.get('question_details', student_data.get('details', []))
        for detail in details:
            if y_pos < 72:
                c.showPage()
                y_pos = height - 72
                
            c.drawString(72, y_pos, f"Q{detail.get('question', '?'):2d}")
            c.drawString(150, y_pos, str(detail.get('detected', 'N/A')))
            c.drawString(220, y_pos, str(detail.get('correct', 'N/A')))
            
            status = detail.get('status', '-')
            if status == 'M':
                c.drawString(290, y_pos, "Multiple")
            else:
                c.drawString(290, y_pos, str(status))
            y_pos -= 15
        
        c.setFont("Helvetica", 8)
        c.drawCentredString(width/2, 30, "Generated by Enhanced OMR Scanner")
        
        c.save()
        
        if not os.path.exists(filepath):
            raise Exception("PDF file was not created")
            
        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            raise Exception(f"PDF file too small ({file_size} bytes)")
        
        print(f"PDF created successfully: {filename}, size: {file_size} bytes")
        return filename
        
    except Exception as e:
        print(f"PDF Export Error: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to export PDF: {str(e)}")

def export_to_excel(student_data: Dict) -> str:
    """Export student results to Excel"""
    try:
        safe_name = "".join(c for c in str(student_data.get('student_name', student_data.get('name', 'Unknown'))) if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"Student_{student_data.get('id', datetime.now().strftime('%H%M%S'))}"
        
        filename = f"omr_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        summary_data = []
        summary_data.append(['Field', 'Value'])
        summary_data.append(['Student Name', str(student_data.get('student_name', student_data.get('name', 'N/A')))])
        summary_data.append(['Registration No', str(student_data.get('reg_no', 'N/A'))])
        summary_data.append(['Class', str(student_data.get('class', 'N/A'))])
        summary_data.append(['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        summary_data.append(['Score', f"{student_data.get('score', 0)}/{student_data.get('total_questions', student_data.get('total', 0))}"])
        summary_data.append(['Percentage', f"{student_data.get('percentage', 0):.1f}%"])
        summary_data.append(['Multiple Marked', str(student_data.get('multiple_answers', 0))])
        summary_data.append(['Bubbles Detected', str(student_data.get('bubbles_detected', 0))])
        
        details_data = []
        details_data.append(['Question', 'Detected Answer', 'Correct Answer', 'Status'])
        details = student_data.get('question_details', student_data.get('details', []))
        for detail in details:
            status = detail.get('status', '-')
            if status == 'M':
                status_text = 'Multiple'
            else:
                status_text = str(status)
            details_data.append([
                f"Q{detail.get('question', '?')}",
                str(detail.get('detected', 'N/A')),
                str(detail.get('correct', 'N/A')),
                status_text
            ])
        
        summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
        details_df = pd.DataFrame(details_data[1:], columns=details_data[0])
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=0)
                details_df.to_excel(writer, sheet_name='Question Details', index=False, startrow=0)
        except Exception as write_error:
            try:
                with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=0)
                    details_df.to_excel(writer, sheet_name='Question Details', index=False, startrow=0)
            except Exception as fallback_error:
                raise Exception(f"Both Excel engines failed: openpyxl - {write_error}, xlsxwriter - {fallback_error}")
        
        if not os.path.exists(filepath):
            raise Exception("Excel file was not created")
            
        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            raise Exception(f"Excel file too small ({file_size} bytes)")
        
        print(f"Excel created successfully: {filename}, size: {file_size} bytes")
        return filename
        
    except Exception as e:
        print(f"Excel Export Error: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to export Excel: {str(e)}")

def export_to_word(student_data: Dict) -> str:
    """Export student results to Word document"""
    try:
        safe_name = "".join(c for c in str(student_data.get('student_name', student_data.get('name', 'Unknown'))) if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"Student_{student_data.get('id', datetime.now().strftime('%H%M%S'))}"
        
        filename = f"omr_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        doc = Document()
        
        title = doc.add_heading('OMR SCAN REPORT', 0)
        title.alignment = 1
        
        doc.add_paragraph('')
        info_heading = doc.add_heading('Student Information', level=2)
        
        p = doc.add_paragraph()
        p.add_run('Name: ').bold = True
        p.add_run(f"{student_data.get('student_name', student_data.get('name', 'N/A'))}\n")
        p.add_run('Registration No: ').bold = True
        p.add_run(f"{student_data.get('reg_no', 'N/A')}\n")
        p.add_run('Class: ').bold = True
        p.add_run(f"{student_data.get('class', 'N/A')}\n")
        p.add_run('Date: ').bold = True
        p.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        p.add_run('Score: ').bold = True
        p.add_run(f"{student_data.get('score', 0)}/{student_data.get('total_questions', student_data.get('total', 0))} ({student_data.get('percentage', 0):.1f}%)\n")
        p.add_run('Multiple Marked: ').bold = True
        p.add_run(f"{student_data.get('multiple_answers', 0)}")
        
        doc.add_paragraph('')
        results_heading = doc.add_heading('Question Details', level=2)
        
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'
        
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Question'
        hdr_cells[1].text = 'Detected'
        hdr_cells[2].text = 'Correct'
        hdr_cells[3].text = 'Status'
        
        for cell in hdr_cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
        
        details = student_data.get('question_details', student_data.get('details', []))
        for detail in details:
            row_cells = table.add_row().cells
            row_cells[0].text = f"Q{detail.get('question', '?')}"
            row_cells[1].text = str(detail.get('detected', 'N/A'))
            row_cells[2].text = str(detail.get('correct', 'N/A'))
            
            status = detail.get('status', '-')
            if status == 'M':
                row_cells[3].text = 'Multiple'
            else:
                row_cells[3].text = str(status)
        
        doc.add_paragraph('')
        footer = doc.add_paragraph('Report generated by Enhanced OMR Scanner')
        footer.alignment = 1
        
        doc.save(filepath)
        
        if not os.path.exists(filepath):
            raise Exception("Word document was not created")
            
        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            raise Exception(f"Word document too small ({file_size} bytes)")
        
        print(f"Word document created successfully: {filename}, size: {file_size} bytes")
        return filename
        
    except Exception as e:
        print(f"Word Export Error: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to export Word: {str(e)}")

def export_to_csv(student_data: Dict) -> str:
    """Export student results to CSV"""
    try:
        safe_name = "".join(c for c in str(student_data.get('student_name', student_data.get('name', 'Unknown'))) if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"Student_{student_data.get('id', datetime.now().strftime('%H%M%S'))}"
        
        filename = f"omr_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        csv_content = []
        
        csv_content.append("OMR SCAN REPORT")
        csv_content.append("")
        csv_content.append("Student Information")
        csv_content.append(f"Name,{student_data.get('student_name', student_data.get('name', 'N/A'))}")
        csv_content.append(f"Registration No,{student_data.get('reg_no', 'N/A')}")
        csv_content.append(f"Class,{student_data.get('class', 'N/A')}")
        csv_content.append(f"Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        csv_content.append(f"Score,{student_data.get('score', 0)}/{student_data.get('total_questions', student_data.get('total', 0))}")
        csv_content.append(f"Percentage,{student_data.get('percentage', 0):.1f}%")
        csv_content.append(f"Multiple Marked,{student_data.get('multiple_answers', 0)}")
        csv_content.append(f"Bubbles Detected,{student_data.get('bubbles_detected', 0)}")
        csv_content.append("")
        csv_content.append("Question Details")
        csv_content.append("Question,Detected,Correct,Status")
        
        details = student_data.get('question_details', student_data.get('details', []))
        for detail in details:
            status = detail.get('status', '-')
            if status == 'M':
                status_text = 'Multiple'
            else:
                status_text = str(status)
            csv_content.append(f"Q{detail.get('question', '?')},{detail.get('detected', 'N/A')},{detail.get('correct', 'N/A')},{status_text}")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            for line in csv_content:
                f.write(line + '\n')
        
        if not os.path.exists(filepath):
            raise Exception("CSV file was not created")
            
        file_size = os.path.getsize(filepath)
        if file_size < 100:
            raise Exception(f"CSV file too small ({file_size} bytes)")
        
        print(f"CSV created successfully: {filename}, size: {file_size} bytes")
        return filename
        
    except Exception as e:
        print(f"CSV Export Error: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to export CSV: {str(e)}")

def export_to_txt(student_data: Dict) -> str:
    """Export student results to TXT"""
    try:
        safe_name = "".join(c for c in str(student_data.get('student_name', student_data.get('name', 'Unknown'))) if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            safe_name = f"Student_{student_data.get('id', datetime.now().strftime('%H%M%S'))}"
        
        filename = f"omr_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        content_lines = []
        content_lines.append("=" * 60)
        content_lines.append("                    OMR SCAN REPORT")
        content_lines.append("=" * 60)
        content_lines.append("")
        content_lines.append("STUDENT INFORMATION:")
        content_lines.append("-" * 20)
        content_lines.append(f"Name:            {student_data.get('student_name', student_data.get('name', 'N/A'))}")
        content_lines.append(f"Registration No: {student_data.get('reg_no', 'N/A')}")
        content_lines.append(f"Class:           {student_data.get('class', 'N/A')}")
        content_lines.append(f"Date:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content_lines.append("")
        content_lines.append("RESULTS SUMMARY:")
        content_lines.append("-" * 16)
        content_lines.append(f"Score:           {student_data.get('score', 0)}/{student_data.get('total_questions', student_data.get('total', 0))}")
        content_lines.append(f"Percentage:      {student_data.get('percentage', 0):.1f}%")
        content_lines.append(f"Multiple Marked: {student_data.get('multiple_answers', 0)}")
        content_lines.append(f"Bubbles Found:   {student_data.get('bubbles_detected', 0)}")
        content_lines.append("")
        content_lines.append("DETAILED RESULTS:")
        content_lines.append("-" * 17)
        content_lines.append("Question | Detected | Correct | Status")
        content_lines.append("-" * 40)
        
        details = student_data.get('question_details', student_data.get('details', []))
        for detail in details:
            status = detail.get('status', '-')
            if status == 'M':
                status_display = 'MULT'
            else:
                status_display = str(status)[:4]
            content_lines.append(f"   Q{detail.get('question', '?'):2d}   |    {str(detail.get('detected', 'N/A'))[:8]:8s} |    {str(detail.get('correct', 'N/A')):1s}    |   {status_display}")
        
        content_lines.append("-" * 40)
        content_lines.append("")
        content_lines.append("Report generated by Enhanced OMR Scanner")
        content_lines.append("=" * 60)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        if not os.path.exists(filepath):
            raise Exception("TXT file was not created")
            
        file_size = os.path.getsize(filepath)
        if file_size < 200:
            raise Exception(f"TXT file too small ({file_size} bytes)")
        
        print(f"TXT created successfully: {filename}, size: {file_size} bytes")
        return filename
        
    except Exception as e:
        print(f"TXT Export Error: {e}")
        traceback.print_exc()
        raise Exception(f"Failed to export TXT: {str(e)}")

# Answer key loading functions
def create_sample_answer_key() -> Dict[int, str]:
    return {1: "B", 2: "D", 3: "A", 4: "C", 5: "B", 6: "A", 7: "C", 8: "D", 9: "B", 10: "A"}

def load_answer_key_from_file_content(content: str, filename: str) -> Dict[int, str]:
    if filename.lower().endswith('.json'):
        data = json.loads(content)
        return {int(k): str(v).strip().upper() for k, v in data.items()}
    elif filename.lower().endswith('.csv'):
        lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
        answer_key: Dict[int, str] = {}
        start = 1 if lines and not lines[0].split(',')[0].strip().isdigit() else 0
        for line in lines[start:]:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 2 and parts[0].isdigit():
                qnum = int(parts[0])
                ans = parts[-1].upper()
                answer_key[qnum] = ans
        return answer_key
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Enhanced OMR Scanner API with Database Export Fixed",
        "version": "8.0.0-DATABASE-EXPORT-FIXED",
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "uploads": os.path.exists(UPLOAD_FOLDER),
            "results": os.path.exists(RESULTS_FOLDER)
        },
        "detection_settings": {
            "bubble_threshold": BUBBLE_THRESHOLD,
            "dynamic_threshold": DYNAMIC_THRESHOLD,
            "relative_threshold": RELATIVE_THRESHOLD
        }
    })

# Database viewer API endpoints - FIXED AND COMPLETE
@app.route('/api/database/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Total students
        cursor.execute('SELECT COUNT(*) FROM students')
        total_students = cursor.fetchone()[0]
        
        # Average score
        cursor.execute('SELECT AVG(percentage) FROM students')
        avg_score = cursor.fetchone()[0] or 0
        
        # Highest and lowest scores
        cursor.execute('SELECT MAX(percentage), MIN(percentage) FROM students')
        scores = cursor.fetchone()
        highest_score = scores[0] or 0
        lowest_score = scores[1] or 0
        
        # Total answer keys
        cursor.execute('SELECT COUNT(*) FROM answer_keys')
        total_keys = cursor.fetchone()[0]
        
        # Date range
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM students')
        dates = cursor.fetchone()
        
        # Class distribution
        cursor.execute('SELECT class, COUNT(*) FROM students GROUP BY class')
        class_distribution = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'total_students': total_students,
            'avg_score': round(avg_score, 2) if avg_score else 0,
            'highest_score': round(highest_score, 2) if highest_score else 0,
            'lowest_score': round(lowest_score, 2) if lowest_score else 0,
            'total_keys': total_keys,
            'date_range': {
                'from': dates[0] if dates[0] else None,
                'to': dates[1] if dates[1] else None
            },
            'class_distribution': [
                {'class': c[0] or 'N/A', 'count': c[1]} for c in class_distribution
            ]
        })
        
    except Exception as e:
        print(f"Database stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/export/csv', methods=['GET'])
def export_database_csv():
    """Export entire database to CSV - FIXED"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        
        # Fetch all student data
        query = """
            SELECT id, name, reg_no, class, score, total, percentage, 
                   timestamp, detected_answers, bubbles_detected, rows_detected
            FROM students
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Parse detected answers and calculate statistics
        def parse_answers(answers_json):
            try:
                answers = json.loads(answers_json) if isinstance(answers_json, str) else answers_json
                multiple_count = sum(1 for v in answers.values() if v == 'MULTIPLE')
                blank_count = sum(1 for v in answers.values() if v in ['N/A', 'BLANK'])
                return multiple_count, blank_count
            except:
                return 0, 0
        
        # Add multiple and blank counts
        df[['multiple_answers', 'blank_answers']] = df['detected_answers'].apply(
            lambda x: pd.Series(parse_answers(x))
        )
        
        # Calculate wrong answers
        df['wrong_answers'] = df['total'] - df['score'] - df['blank_answers'] - df['multiple_answers']
        df['wrong_answers'] = df['wrong_answers'].clip(lower=0)  # Ensure no negative values
        
        # Drop the JSON column for cleaner export
        df = df.drop('detected_answers', axis=1)
        
        # Rename columns for better readability
        df.columns = ['ID', 'Student Name', 'Reg No', 'Class', 'Score', 'Total Questions', 
                     'Percentage', 'Timestamp', 'Bubbles Detected', 'Rows Detected',
                     'Multiple Answers', 'Blank Answers', 'Wrong Answers']
        
        # Create CSV
        filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        # Ensure directory exists
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        
        conn.close()
        
        # Verify file was created
        if not os.path.exists(filepath):
            raise Exception("CSV file was not created")
        
        return send_file(filepath, 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name=filename)
        
    except Exception as e:
        print(f"CSV Export Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/export/excel', methods=['GET'])
def export_database_excel():
    """Export entire database to Excel with multiple sheets - FIXED"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        
        # Fetch all student data
        students_query = """
            SELECT id, name, reg_no, class, score, total, percentage, 
                   timestamp, detected_answers, bubbles_detected, rows_detected
            FROM students
            ORDER BY timestamp DESC
        """
        
        df_students = pd.read_sql_query(students_query, conn)
        
        if len(df_students) == 0:
            return jsonify({'error': 'No data to export'}), 400
        
        # Parse detected answers and calculate statistics
        def parse_answers(answers_json):
            try:
                answers = json.loads(answers_json) if isinstance(answers_json, str) else answers_json
                multiple_count = sum(1 for v in answers.values() if v == 'MULTIPLE')
                blank_count = sum(1 for v in answers.values() if v in ['N/A', 'BLANK'])
                return multiple_count, blank_count
            except:
                return 0, 0
        
        # Add multiple and blank counts
        df_students[['multiple_answers', 'blank_answers']] = df_students['detected_answers'].apply(
            lambda x: pd.Series(parse_answers(x))
        )
        
        # Calculate wrong answers
        df_students['wrong_answers'] = df_students['total'] - df_students['score'] - df_students['blank_answers'] - df_students['multiple_answers']
        df_students['wrong_answers'] = df_students['wrong_answers'].clip(lower=0)
        
        # Drop the JSON column for cleaner export
        df_students = df_students.drop('detected_answers', axis=1)
        
        # Rename columns
        df_students.columns = ['ID', 'Student Name', 'Reg No', 'Class', 'Score', 'Total Questions', 
                              'Percentage', 'Timestamp', 'Bubbles Detected', 'Rows Detected',
                              'Multiple Answers', 'Blank Answers', 'Wrong Answers']
        
        # Create summary statistics
        summary_data = []
        summary_data.append(['Total Students', len(df_students)])
        summary_data.append(['Average Score', f"{df_students['Percentage'].mean():.2f}%"])
        summary_data.append(['Highest Score', f"{df_students['Percentage'].max():.2f}%"])
        summary_data.append(['Lowest Score', f"{df_students['Percentage'].min():.2f}%"])
        summary_data.append(['Total Multiple Answers', df_students['Multiple Answers'].sum()])
        
        # Class-wise statistics
        class_stats = df_students.groupby('Class').agg({
            'Student Name': 'count',
            'Percentage': 'mean'
        }).round(2)
        class_stats.columns = ['Student Count', 'Average Percentage']
        
        # Create Excel file with multiple sheets
        filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        # Ensure directory exists
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write summary sheet
            df_summary = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write all students data
            df_students.to_excel(writer, sheet_name='All Students', index=False)
            
            # Write class statistics
            class_stats.to_excel(writer, sheet_name='Class Statistics')
            
            # Add date-wise statistics
            df_students['Date'] = pd.to_datetime(df_students['Timestamp']).dt.date
            date_stats = df_students.groupby('Date').agg({
                'Student Name': 'count',
                'Percentage': 'mean'
            }).round(2)
            date_stats.columns = ['Students Scanned', 'Average Score']
            date_stats.to_excel(writer, sheet_name='Daily Statistics')
        
        conn.close()
        
        # Verify file was created
        if not os.path.exists(filepath):
            raise Exception("Excel file was not created")
        
        return send_file(filepath,
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name=filename)
        
    except Exception as e:
        print(f"Excel Export Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/backup', methods=['POST'])
def backup_database():
    """Create a backup of the database"""
    try:
        backup_filename = f"omr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = os.path.join(RESULTS_FOLDER, backup_filename)
        
        # Copy the database file
        import shutil
        shutil.copy2(DATABASE_FILE, backup_path)
        
        # Get file size
        file_size = os.path.getsize(backup_path)
        
        return jsonify({
            'message': 'Database backup created successfully',
            'filename': backup_filename,
            'size': file_size,
            'path': backup_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Batch operations for database management
@app.route('/api/database/delete-multiple', methods=['POST'])
def delete_multiple_records():
    """Delete multiple student records at once"""
    try:
        data = request.get_json()
        ids_to_delete = data.get('ids', [])
        
        if not ids_to_delete:
            return jsonify({'error': 'No IDs provided'}), 400
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Create placeholders for SQL query
        placeholders = ','.join('?' * len(ids_to_delete))
        query = f'DELETE FROM students WHERE id IN ({placeholders})'
        
        cursor.execute(query, ids_to_delete)
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': f'Successfully deleted {deleted_count} records',
            'deleted_count': deleted_count,
            'deleted_ids': ids_to_delete
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Keep all existing API routes
@app.route('/api/get-answer-key', methods=['GET'])
def get_answer_key():
    """Get current answer key with display"""
    try:
        answer_key = get_current_answer_key()
        if not answer_key:
            return jsonify({
                "message": "No answer key loaded",
                "answer_key": {},
                "total_questions": 0,
                "display": "No answer key available"
            })
        
        display_text = "Current Answer Key:\n"
        for q_num in sorted(answer_key.keys()):
            display_text += f"Q{q_num}: {answer_key[q_num]}  "
            if q_num % 5 == 0:
                display_text += "\n"
        
        return jsonify({
            "message": f"Answer key loaded with {len(answer_key)} questions",
            "answer_key": answer_key,
            "total_questions": len(answer_key),
            "display": display_text.strip()
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get answer key: {str(e)}"}), 500

@app.route('/api/scan-answer-key', methods=['POST'])
def scan_answer_key():
    """Scan answer key from camera/uploaded image or mobile camera"""
    try:
        image = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                os.remove(filepath)
        
        elif request.json and 'url' in request.json:
            camera_url = request.json['url']
            resp = urllib.request.urlopen(camera_url)
            image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        elif request.json and 'image' in request.json:
            base64_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
            image_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not read image"}), 400
        
        temp_scanner = OMRScanner({})
        scanned_answer_key = temp_scanner.scan_answer_key_from_image(image)
        
        key_name = f"Scanned_Key_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_answer_key(key_name, scanned_answer_key, set_as_current=True)
        
        return jsonify({
            "message": f"Answer key scanned and saved successfully with {len(scanned_answer_key)} questions",
            "answer_key": scanned_answer_key
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to scan answer key: {str(e)}"}), 500

@app.route('/api/load-answer-key', methods=['POST'])
def load_answer_key():
    """Load answer key from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        content = file.read().decode('utf-8')
        answer_key_data = load_answer_key_from_file_content(content, file.filename)
        
        key_name = f"Loaded_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_answer_key(key_name, answer_key_data, set_as_current=True)
        
        return jsonify({
            "message": f"Answer key loaded and saved successfully with {len(answer_key_data)} questions",
            "answer_key": answer_key_data
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to load answer key: {str(e)}"}), 500

@app.route('/api/scan-single', methods=['POST'])
def scan_single_omr():
    """Scan a single OMR sheet with student details - supports mobile camera"""
    try:
        answer_key = get_current_answer_key()
        if not answer_key:
            return jsonify({"error": "No answer key loaded. Please load or scan an answer key first."}), 400

        image = None
        student_name = 'Unknown'
        reg_no = 'N/A'
        student_class = 'N/A'
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image = cv2.imread(filepath)
                os.remove(filepath)
                
                student_name = request.form.get('student_name', 'Unknown')
                reg_no = request.form.get('reg_no', 'N/A')
                student_class = request.form.get('class', 'N/A')
        
        elif request.json:
            if 'image' in request.json:
                base64_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
                image_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            student_name = request.json.get('student_name', 'Unknown')
            reg_no = request.json.get('reg_no', 'N/A')
            student_class = request.json.get('class', 'N/A')
        
        if image is None:
            return jsonify({"error": "Could not read image"}), 400

        scanner = OMRScanner(answer_key)
        results = scanner.scan_sheet_from_image(image)
        
        student_data = {
            'name': student_name,
            'reg_no': reg_no,
            'class': student_class,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        student_id = save_student_result(student_data)
        
        correct_answers, wrong_answers, blank_answers, multiple_answers = calculate_answer_statistics(results['details'])
        
        return jsonify({
            "message": "OMR sheet scanned and saved successfully",
            "id": student_id,
            "student_id": student_id,
            "student_name": student_name,
            "reg_no": reg_no,
            "class": student_class,
            "score": results["score"],
            "total": results["total"],
            "total_questions": results["total"],
            "percentage": results["percentage"],
            "detected_answers": results["detected_answers"],
            "question_details": results["details"],
            "details": results["details"],
            "bubbles_detected": results["bubbles_detected"],
            "rows_detected": results["rows_detected"],
            "timestamp": student_data['timestamp'],
            "answer_key": answer_key,
            "correct_answers": correct_answers,
            "wrong_answers": wrong_answers,
            "blank_answers": blank_answers,
            "multiple_answers": multiple_answers
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the OMR sheet: {str(e)}"}), 500

@app.route('/api/scan-multiple', methods=['POST'])
def scan_multiple_omr():
    """Scan multiple OMR sheets with individual student details"""
    try:
        answer_key = get_current_answer_key()
        if not answer_key:
            return jsonify({"error": "No answer key loaded"}), 400

        if request.files:
            files = request.files.getlist('files')
            if not files:
                return jsonify({"error": "No files provided"}), 400
            
            student_names = request.form.getlist('student_names[]') or []
            reg_nos = request.form.getlist('reg_nos[]') or []
            classes = request.form.getlist('classes[]') or []
            
        elif request.json:
            images_data = request.json.get('images', [])
            student_details = request.json.get('student_details', [])
            
            if not images_data:
                return jsonify({"error": "No images provided"}), 400
            
            files = []
            student_names = []
            reg_nos = []
            classes = []
            
            for i, img_data in enumerate(images_data):
                base64_data = img_data.split(',')[1] if ',' in img_data else img_data
                image_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    temp_filename = f"temp_multiple_{i}_{datetime.now().strftime('%H%M%S')}.jpg"
                    temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
                    cv2.imwrite(temp_filepath, image)
                    files.append({'path': temp_filepath, 'name': f"sheet_{i+1}.jpg"})
                
                if i < len(student_details):
                    student_names.append(student_details[i].get('name', f'Student_{i+1}'))
                    reg_nos.append(student_details[i].get('reg_no', f'REG_{i+1:03d}'))
                    classes.append(student_details[i].get('class', 'Batch_Scan'))
                else:
                    student_names.append(f'Student_{i+1}')
                    reg_nos.append(f'REG_{i+1:03d}')
                    classes.append('Batch_Scan')
        
        results = []
        scanner = OMRScanner(answer_key)

        if request.files:
            for i, file in enumerate(files):
                if file.filename == '' or not allowed_file(file.filename):
                    continue

                try:
                    filename = secure_filename(f"batch_{i}_{file.filename}")
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)

                    image = cv2.imread(filepath)
                    if image is not None:
                        scan_result = scanner.scan_sheet_from_image(image)
                        
                        student_name = student_names[i] if i < len(student_names) else f'Student_{i+1}'
                        reg_no = reg_nos[i] if i < len(reg_nos) else f'REG_{i+1:03d}'
                        student_class = classes[i] if i < len(classes) else 'Batch_Scan'
                        
                        student_data = {
                            'name': student_name,
                            'reg_no': reg_no,
                            'class': student_class,
                            'results': scan_result,
                            'filename': file.filename,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        student_id = save_student_result(student_data)
                        
                        correct_answers, wrong_answers, blank_answers, multiple_answers = calculate_answer_statistics(scan_result['details'])
                        
                        result_data = {
                            'id': student_id,
                            'student_name': student_name,
                            'reg_no': reg_no,
                            'class': student_class,
                            'filename': file.filename,
                            'timestamp': student_data['timestamp'],
                            'score': scan_result['score'],
                            'total_questions': scan_result['total'],
                            'percentage': scan_result['percentage'],
                            'detected_answers': scan_result['detected_answers'],
                            'question_details': scan_result['details'],
                            'bubbles_detected': scan_result['bubbles_detected'],
                            'rows_detected': scan_result['rows_detected'],
                            'correct_answers': correct_answers,
                            'wrong_answers': wrong_answers,
                            'blank_answers': blank_answers,
                            'multiple_answers': multiple_answers
                        }
                        
                        results.append(result_data)

                    os.remove(filepath)

                except Exception as e:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    continue
        
        elif request.json:
            for i, file_info in enumerate(files):
                try:
                    image = cv2.imread(file_info['path'])
                    if image is not None:
                        scan_result = scanner.scan_sheet_from_image(image)
                        
                        student_data = {
                            'name': student_names[i],
                            'reg_no': reg_nos[i],
                            'class': classes[i],
                            'results': scan_result,
                            'filename': file_info['name'],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        student_id = save_student_result(student_data)
                        
                        correct_answers, wrong_answers, blank_answers, multiple_answers = calculate_answer_statistics(scan_result['details'])
                        
                        result_data = {
                            'id': student_id,
                            'student_name': student_names[i],
                            'reg_no': reg_nos[i],
                            'class': classes[i],
                            'filename': file_info['name'],
                            'timestamp': student_data['timestamp'],
                            'score': scan_result['score'],
                            'total_questions': scan_result['total'],
                            'percentage': scan_result['percentage'],
                            'detected_answers': scan_result['detected_answers'],
                            'question_details': scan_result['details'],
                            'bubbles_detected': scan_result['bubbles_detected'],
                            'rows_detected': scan_result['rows_detected'],
                            'correct_answers': correct_answers,
                            'wrong_answers': wrong_answers,
                            'blank_answers': blank_answers,
                            'multiple_answers': multiple_answers
                        }
                        
                        results.append(result_data)

                    os.remove(file_info['path'])

                except Exception as e:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                    continue

        return jsonify({
            "message": f"Processed and saved {len(results)} OMR sheets successfully",
            "results": results,
            "total_processed": len(results),
            "answer_key": answer_key
        })

    except Exception as e:
        return jsonify({"error": f"Batch processing error: {str(e)}"}), 500

@app.route('/api/scan-camera', methods=['POST'])
def scan_from_ip_camera():
    """Scan OMR sheet from IP camera or mobile camera with student details"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        answer_key = get_current_answer_key()
        if not answer_key:
            return jsonify({"error": "No answer key loaded"}), 400

        student_name = data.get('student_name', 'Unknown')
        reg_no = data.get('reg_no', 'N/A')
        student_class = data.get('class', 'N/A')
        
        image = None
        
        if 'url' in data:
            camera_url = data['url'].strip()
            try:
                resp = urllib.request.urlopen(camera_url)
                image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Failed to capture from camera: {str(e)}"}), 400
        
        elif 'image' in data:
            try:
                base64_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
                image_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Failed to process mobile camera image: {str(e)}"}), 400
        
        if image is None:
            return jsonify({"error": "Could not decode image from camera"}), 400

        scanner = OMRScanner(answer_key)
        results = scanner.scan_sheet_from_image(image)
        
        student_data = {
            'name': student_name,
            'reg_no': reg_no,
            'class': student_class,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        student_id = save_student_result(student_data)
        
        correct_answers, wrong_answers, blank_answers, multiple_answers = calculate_answer_statistics(results['details'])

        return jsonify({
            "message": "OMR sheet scanned from camera and saved successfully",
            "id": student_id,
            "student_id": student_id,
            "student_name": student_name,
            "reg_no": reg_no,
            "class": student_class,
            "score": results["score"],
            "total": results["total"],
            "total_questions": results["total"],
            "percentage": results["percentage"],
            "detected_answers": results["detected_answers"],
            "question_details": results["details"],
            "details": results["details"],
            "bubbles_detected": results["bubbles_detected"],
            "rows_detected": results["rows_detected"],
            "timestamp": student_data['timestamp'],
            "answer_key": answer_key,
            "correct_answers": correct_answers,
            "wrong_answers": wrong_answers,
            "blank_answers": blank_answers,
            "multiple_answers": multiple_answers
        })

    except Exception as e:
        return jsonify({"error": f"Camera scan error: {str(e)}"}), 500

@app.route('/api/export/<format_type>/<int:result_id>', methods=['GET'])
def export_result(format_type, result_id):
    """Export results in various formats"""
    try:
        print(f"Export request received: format={format_type}, result_id={result_id}")
        
        all_results = get_all_student_results()
        
        student_data = None
        for result in all_results:
            if result.get('id') == result_id:
                student_data = result
                break
        
        if not student_data:
            print(f"Result with ID {result_id} not found")
            return jsonify({"error": f"Result with ID {result_id} not found", "success": False}), 404
        
        print(f"Found student data for export: {student_data.get('student_name', 'Unknown')}")
        
        export_functions = {
            'pdf': export_to_pdf,
            'word': export_to_word,
            'csv': export_to_csv,
            'xlsx': export_to_excel,
            'excel': export_to_excel,
            'txt': export_to_txt
        }
        
        if format_type not in export_functions:
            return jsonify({
                "error": f"Unsupported format '{format_type}'. Supported formats: {list(export_functions.keys())}", 
                "success": False
            }), 400
        
        print(f"Exporting result {result_id} to {format_type} format...")
        
        filename = export_functions[format_type](student_data)
        
        print(f"Export successful: {filename}")
        
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(file_path):
            raise Exception(f"Export file was not created: {filename}")
        
        file_size = os.path.getsize(file_path)
        
        return jsonify({
            "message": f"Results exported to {format_type.upper()} successfully",
            "filename": filename,
            "download_url": f"/api/download/{filename}",
            "student_name": student_data.get('student_name', 'Unknown'),
            "export_format": format_type,
            "file_size": file_size,
            "success": True
        })
        
    except Exception as e:
        print(f"Export Error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Export error: {str(e)}", "success": False}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download exported file"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({"error": f"File not found: {filename}"}), 404
        
        file_size = os.path.getsize(file_path)
        print(f"Serving file: {filename}, size: {file_size} bytes")
        
        if file_size == 0:
            print(f"File is empty: {filename}")
            return jsonify({"error": f"File is empty: {filename}"}), 500
        
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv',
            '.txt': 'text/plain'
        }
        
        file_ext = os.path.splitext(filename)[1].lower()
        mimetype = content_types.get(file_ext, 'application/octet-stream')
        
        print(f"Serving file with mimetype: {mimetype}")
        
        return send_from_directory(
            RESULTS_FOLDER, 
            filename, 
            as_attachment=True,
            mimetype=mimetype,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Download error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Download error: {str(e)}"}), 500

@app.route('/api/results', methods=['GET'])
def get_all_results():
    """Get all stored results from database"""
    try:
        results = get_all_student_results()
        print(f"Retrieved {len(results)} results from database")
        return jsonify({
            "results": results,
            "total": len(results),
            "message": f"Found {len(results)} scan results in database"
        })
    except Exception as e:
        print(f"Error retrieving results: {e}")
        return jsonify({"error": f"Error retrieving results: {str(e)}"}), 500

@app.route('/api/results/<int:result_id>', methods=['DELETE'])
def delete_single_result(result_id):
    """Delete a single result"""
    try:
        success = delete_student_result(result_id)
        
        if success:
            return jsonify({
                "message": f"Result {result_id} deleted successfully",
                "deleted_id": result_id,
                "success": True
            })
        else:
            return jsonify({
                "error": f"Result {result_id} not found",
                "success": False
            }), 404
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to delete result: {str(e)}",
            "success": False
        }), 500

@app.route('/api/results', methods=['DELETE'])
def clear_all_results():
    """Clear all results"""
    try:
        deleted_count = clear_all_student_results()
        
        return jsonify({
            "message": f"All {deleted_count} results cleared successfully",
            "deleted_count": deleted_count,
            "success": True
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to clear results: {str(e)}",
            "success": False
        }), 500

@app.route('/api/set-sample-answer-key', methods=['POST'])
def set_sample_answer_key():
    """Set sample answer key for testing"""
    try:
        sample_key = create_sample_answer_key()
        save_answer_key("Sample_Answer_Key", sample_key, set_as_current=True)
        
        return jsonify({
            "message": f"Sample answer key set and saved with {len(sample_key)} questions",
            "answer_key": sample_key
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to set sample answer key: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        answer_key = get_current_answer_key()
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM students')
        total_students = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM answer_keys')
        total_answer_keys = cursor.fetchone()[0]
        conn.close()
        
        return jsonify({
            "system_status": "ready",
            "database_connected": True,
            "answer_key_loaded": answer_key is not None,
            "answer_key_questions": len(answer_key) if answer_key else 0,
            "total_results": total_students,
            "total_answer_keys": total_answer_keys,
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER),
            "results_folder_exists": os.path.exists(RESULTS_FOLDER),
            "database_file_exists": os.path.exists(DATABASE_FILE),
            "server_time": datetime.now().isoformat(),
            "version": "8.0.0-DATABASE-EXPORT-FIXED",
            "detection_settings": {
                "bubble_threshold": BUBBLE_THRESHOLD,
                "dynamic_threshold": DYNAMIC_THRESHOLD,
                "relative_threshold": RELATIVE_THRESHOLD
            },
            "features": [
                "FIXED: Database export functionality working",
                "FIXED: Multiple bubble detection working correctly",
                "Database viewer with filtering and sorting",
                "Bulk delete and individual delete options",
                "CSV and Excel export for entire database",
                "Dynamic threshold calculation for better accuracy"
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Status check error: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

if __name__ == '__main__':
    print("Starting Enhanced OMR Scanner API v8.0.0")
    print("=" * 80)
    print("Features:")
    print("   ✅ Frontend serving routes added")
    print("   ✅ Database export functionality")
    print("   ✅ Multiple bubble detection")
    print("   ✅ Delete functionality")
    print("   ✅ Database viewer support")
    print("")
    print("Available endpoints:")
    print("- GET  / (serves React frontend)")
    print("- GET  /api/health (API health check)")
    print("- GET  /api/database/stats (database statistics)")
    print("- GET  /api/database/export/csv (export all to CSV)")
    print("- GET  /api/database/export/excel (export all to Excel)")
    print("")
    print(f"Server starting on http://localhost:5000")
    print("=" * 80)
    
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"Created directory: {folder}")
    
    # Use PORT from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Set debug=False for production
    app.run(debug=False, host='0.0.0.0', port=port)