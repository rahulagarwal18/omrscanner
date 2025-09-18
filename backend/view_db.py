import sqlite3
import json
from datetime import datetime

# Connect to database
conn = sqlite3.connect('omr_results.db')
cursor = conn.cursor()

print("="*80)
print("DATABASE VIEWER FOR OMR SCANNER")
print("="*80)

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in database:")
for table in tables:
    print(f"  • {table[0]}")

print("\n" + "="*80)
print("STUDENTS TABLE SUMMARY:")
print("-"*80)

# Get students data with better formatting
cursor.execute("""
    SELECT id, name, reg_no, class, score, total, percentage, timestamp 
    FROM students 
    ORDER BY id DESC 
    LIMIT 10
""")
students = cursor.fetchall()

if students:
    print(f"{'ID':<5} {'Name':<20} {'Reg No':<15} {'Class':<10} {'Score':<10} {'Percentage':<10} {'Date'}")
    print("-"*80)
    for student in students:
        id, name, reg_no, class_name, score, total, percentage, timestamp = student
        date = timestamp[:10] if timestamp else "N/A"
        print(f"{id:<5} {name[:20]:<20} {reg_no[:15]:<15} {class_name[:10]:<10} {score}/{total:<8} {percentage:>6.1f}%     {date}")
else:
    print("No student records found.")

print("\n" + "="*80)
print("ANSWER KEYS TABLE:")
print("-"*80)

# Get answer keys with better formatting
cursor.execute("""
    SELECT id, name, answer_key, created_at, is_current 
    FROM answer_keys 
    ORDER BY id DESC
""")
keys = cursor.fetchall()

if keys:
    for key in keys:
        id, name, answer_key_json, created_at, is_current = key
        answer_key = json.loads(answer_key_json)
        status = "✓ CURRENT" if is_current else "  "
        
        print(f"\nID: {id} | Name: {name} | {status}")
        print(f"Created: {created_at}")
        print(f"Answer Key: ", end="")
        
        # Print answer key in a readable format
        for q_num in sorted([int(k) for k in answer_key.keys()]):
            print(f"Q{q_num}:{answer_key[str(q_num)]} ", end="")
            if q_num % 10 == 0:
                print()  # New line every 10 questions
        print()
else:
    print("No answer keys found.")

print("\n" + "="*80)
print("DATABASE STATISTICS:")
print("-"*80)

# Get statistics
cursor.execute("SELECT COUNT(*) FROM students")
total_students = cursor.fetchone()[0]
print(f"Total Students Scanned: {total_students}")

if total_students > 0:
    cursor.execute("SELECT AVG(percentage) FROM students")
    avg_score = cursor.fetchone()[0]
    print(f"Average Score: {avg_score:.2f}%")
    
    cursor.execute("SELECT MAX(percentage) FROM students")
    max_score = cursor.fetchone()[0]
    print(f"Highest Score: {max_score:.2f}%")
    
    cursor.execute("SELECT MIN(percentage) FROM students")
    min_score = cursor.fetchone()[0]
    print(f"Lowest Score: {min_score:.2f}%")

cursor.execute("SELECT COUNT(*) FROM answer_keys")
total_keys = cursor.fetchone()[0]
print(f"Total Answer Keys: {total_keys}")

cursor.execute("SELECT COUNT(*) FROM answer_keys WHERE is_current = 1")
current_key = cursor.fetchone()[0]
if current_key > 0:
    cursor.execute("SELECT name FROM answer_keys WHERE is_current = 1")
    current_key_name = cursor.fetchone()[0]
    print(f"Current Active Key: {current_key_name}")

conn.close()

print("\n" + "="*80)
print("Database viewing complete!")