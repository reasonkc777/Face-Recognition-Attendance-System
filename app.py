import os
import cv2
import sqlite3
import numpy as np
import json
import base64
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session,Response
from keras_facenet import FaceNet
import csv
from io import StringIO

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "face@123"

# Database and Directory Constants
DATABASE = "attendance.db"
USER_DATA_DIR = "user_data"

# Ensure directories exist
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Initialize FaceNet
embedder = FaceNet()

# Database Connection Helper
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Detect Faces Using Haar Cascade
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Initialize Database with Tables 
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create database tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY, 
        first_name TEXT NOT NULL, 
        last_name TEXT NOT NULL,
        batch TEXT NOT NULL,
        course TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    );''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS student_embedding (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        embedding BLOB,
        face_image TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
        attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        admin_id TEXT NOT NULL,
        password TEXT NOT NULL
    );''')
    
    conn.commit()
    conn.close()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register_student.html")

@app.route("/register_student", methods=["GET","POST"])
def register_student():
    if request.method == "GET":
        return render_template("register_student.html")
    elif request.method=="POST":
        try:
            # Get form data
            first_name = request.form.get("first_name")
            last_name = request.form.get("last_name")
            batch = request.form.get("batch")
            course = request.form.get("course")
            student_id = request.form.get("student_id")
            phone = request.form.get("phone")
            email = request.form.get("email")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")

            # Validation
            if not all([first_name, last_name, batch, course, student_id, phone, email, password, confirm_password]):
                flash("All fields are required!", "error")
                return redirect(url_for("register_student"))

            if password != confirm_password:
                flash("Passwords do not match!", "error")
                return redirect(url_for("register_student"))

            # Insert student data into the `students` table
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO students (student_id, first_name, last_name, batch, course, phone, email, password,confirm_password) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)",
                (student_id, first_name, last_name, batch, course, phone, email, password,confirm_password)
            )
            conn.commit()
            conn.close()

            flash("Student registered successfully! Capture face data next.", "success")
            return redirect(url_for("home"))

        except sqlite3.IntegrityError:
            flash("Student ID or email already exists.", "error")
            return redirect(url_for("register_student"))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for("register_student"))

# Capture Student Face
@app.route("/capture_student", methods=["POST"])
def capture_student_face():
    try:
        student_id = request.form.get("student_id")
        if not student_id:
            return jsonify({"message": "Error: Student ID is required"}), 400

        folder_path = os.path.join(USER_DATA_DIR, student_id)
        os.makedirs(folder_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"message": "Error: Camera not accessible"}), 400

        captured_count = 0
        while captured_count < 10:
            ret, frame = cap.read()
            if not ret:
                return jsonify({"message": "Error: Failed to capture frame"}), 400

            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                file_path = os.path.join(folder_path, f"{student_id}_{captured_count}.jpg")
                cv2.imwrite(file_path, frame[y:y+h, x:x+w])
                captured_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if captured_count == 0:
            return jsonify({"message": "No faces detected"}), 400

        embeddings = []
        photo_img = None
        for idx, file in enumerate(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            resized_image = cv2.resize(image, (160, 160))
            resized_image = np.expand_dims(resized_image, axis=0)
            embedding = embedder.embeddings(resized_image)[0]
            embeddings.append(embedding)

            if idx == 0:
                _, buffer = cv2.imencode(".jpg", image)
                photo_img = base64.b64encode(buffer).decode("utf-8")

        if not embeddings:
            return jsonify({"message": "No valid embeddings generated."}), 400

        average_embedding = json.dumps(np.mean(embeddings, axis=0).tolist())

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO student_embedding (student_id, embedding, face_image) VALUES (?, ?, ?)",
            (student_id, average_embedding, photo_img)
        )
        conn.commit()
        conn.close()

        return jsonify({"message": f"Captured {captured_count} images successfully!"}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/login1", methods=["GET", "POST"])
def login1():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")
        profile = request.form.get("profile")  # "student" or "admin"

        # Determine table and role-specific keys
        table = "students" if profile == "student" else "admin"
        id_field = "student_id" if profile == "student" else "admin_id"
        name_field = "first_name" if profile == "student" else "admin_name"

        # Query the appropriate table
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {table} WHERE {id_field} = ? AND password = ?",
                (user_id, password)
            )
            user = cursor.fetchone()  # Fetch one record
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for("login"))
        finally:
            conn.close()

        if user:
            # Store user details in session
            session["user_id"] = user[id_field]  # Access field by column name
            session["user_name"] = user[name_field]  # Access field by column name
            session["profile"] = profile

            # Redirect to the appropriate dashboard
            if profile == "student":
                return redirect(url_for("student_dashboard"))
            else:
                return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials, please try again.", "error")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/student_dashboard")
def student_dashboard():
    # Check if the user is logged in and is a student
    if "user_id" not in session or session.get("profile") != "student":
        flash("Unauthorized access. Please log in as a student.", "error")
        return redirect(url_for("login"))
    student_id = session.get("user_id")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) as present_days 
        FROM attendance 
        WHERE student_id = ?
        """,
        (student_id,)
    )
    result = cursor.fetchone()
    conn.close()

    # If no attendance records are found, set present_days to 0
    present_days = result["present_days"] if result else 0

    # Render the student dashboard template with the attendance data
    return render_template(
        "student_dashboard.html", 
        first_name=session["user_name"], 
        present_days=present_days
    )

@app.route("/admin_dashboard")
def admin_dashboard():
    # Check if the user is logged in and is an admin
    if "user_id" not in session or session.get("profile") != "admin":
        flash("Unauthorized access. Please log in as an admin.", "error")
        return redirect(url_for("login"))

    # Render the admin dashboard
    return render_template("admin_dashboard.html", admin_name=session["user_name"])

@app.route("/student_attendance_sheet")
def student_attendance_sheet():
    if "user_id" not in session:
        return redirect(url_for("login"))

    student_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    # SQL query to join attendance with students table on student_id
    query = """
    SELECT 
        a.attendance_id, a.timestamp, s.first_name, s.last_name, s.batch, s.student_id, s.course
    FROM 
        attendance a
    JOIN 
        students s ON a.student_id = s.student_id
    WHERE 
        a.student_id = ?
    """
    cursor.execute(query, (student_id,))
    attendance_records = [dict(row) for row in cursor.fetchall()]
    conn.close()
     # Convert the timestamp from string to datetime object

    for record in attendance_records:
        if isinstance(record['timestamp'], str):
            record['timestamp'] = datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S')
    if not attendance_records:
        flash("No attendance records found for this student.", "error")
    return render_template("student_attendance_sheet.html", attendance_records=attendance_records)


@app.route("/take_attendance", methods=["POST"])
def take_attendance():
    try:
        print("Starting attendance process...")  # Debugging log
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash("Unable to access the camera. Please check your setup.", "error")
            return redirect(url_for("student_dashboard"))

        ret, frame = cap.read()
        if not ret:
            flash("Failed to capture frame from the camera.", "error")
            return redirect(url_for("student_dashboard"))

        print("Capturing faces...")  # Debugging log
        faces = detect_faces(frame)
        if len(faces) == 0:
            flash("No faces detected. Please try again.", "error")
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for("student_dashboard"))

        embedding = None
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_img, (160, 160))
            resized_face = np.expand_dims(resized_face, axis=0)
            embedding = embedder.embeddings(resized_face)[0]

        conn = get_db_connection()  # Ensure get_db_connection() is properly defined
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, embedding FROM student_embedding")
        rows = cursor.fetchall()

        # Manual cosine similarity
        def cosine_similarity_manual(vector1, vector2):
            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        for row in rows:
            stored_embedding = json.loads(row["embedding"])  # Assuming embeddings are stored in JSON format
            stored_embedding = np.array(stored_embedding)

            # Use manual cosine similarity function
            similarity = cosine_similarity_manual(embedding, stored_embedding)

            if similarity > 0.7:
                student_id = row["student_id"]

                # Retrieve student_name from the students table
                cursor.execute("SELECT first_name FROM students WHERE student_id = ?", (student_id,))
                student = cursor.fetchone()
                if not student:
                    flash(f"Student record not found for ID: {student_id}", "error")
                    break

                student_name = student["first_name"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO attendance (student_id, student_name, timestamp) VALUES (?, ?, ?)",
                    (student_id, student_name, timestamp)
                )
                conn.commit()
                flash(f"Attendance marked successfully for student {student_name}!", "success")
                print(f"Attendance recorded for {student_name}")  # Debugging log
                break
        else:
            flash("Face not recognized. Please try again.", "error")

        conn.close()
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("student_dashboard"))

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for("student_dashboard"))


@app.template_filter('to_datetime')
def to_datetime(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S') if isinstance(value, str) else value

@app.template_filter('format_date')
def format_date(value):
    return value.strftime('%Y-%m-%d') if value else 'N/A'

@app.route("/student_details", methods=["GET"])
def student_details():
    if "user_id" not in session:
        return redirect(url_for("login"))

    student_id = session["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query to get student details by student_id
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()

    # Check if student record exists
    if student is None:
        flash("Student details not found.", "error")
        return redirect(url_for("student_attendance"))

    # Pass the student details to the template
    return render_template("student_details.html", student=student)

@app.route('/manage_students', methods=['GET'])
def manage_students():
    search_query = request.args.get('search_query', '').strip()
    delete_id = request.args.get('delete_id')

    conn = get_db_connection()

    # If delete_id is provided, delete the student record
    if delete_id:
        conn.execute('DELETE FROM students WHERE student_id = ?', (delete_id,))
        conn.commit()
        conn.close()
        return redirect(url_for('manage_students'))

    # Fetch students based on the search query
    if search_query:
        students = conn.execute('''
            SELECT * FROM students
            WHERE first_name LIKE ? OR last_name LIKE ? OR batch LIKE ? OR course LIKE ?
        ''', (f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%')).fetchall()
    else:
        # Fetch all students if no search query
        students = conn.execute('SELECT * FROM students').fetchall()

    conn.close()

    # Render the template with the student data
    return render_template('manage_students.html', students=students, search_query=search_query)

@app.route("/admin_attendance")
def admin_attendance():
    return render_template("admin_attendance.html")

@app.route('/attendance', methods=['GET','POST'])
def attendance():
    search_query = request.args.get('search_query', '').strip()

    conn = get_db_connection()

    # Fetch attendance records
    if search_query:
        # Filter attendance based on student name, batch, or date
        attendance_records = conn.execute('''
            SELECT a.attendance_id, s.first_name, s.last_name,s.batch, a.timestamp
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE s.first_name LIKE ? OR s.last_name LIKE ?OR s.batch LIKE ? OR a.timestamp LIKE ?
        ''', (f'%{search_query}%', f'%{search_query}%', f'%{search_query}%',f'%{search_query}%')).fetchall()
    else:
        # Fetch all attendance records
        attendance_records = conn.execute('''
            SELECT a.attendance_id, s.first_name, s.last_name,S.batch, a.timestamp
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
        ''').fetchall()
    # Handle the POST request to download CSV
    if request.method == 'POST' and 'download_csv' in request.form:
        # Create an in-memory file for the CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Write CSV header
        writer.writerow(['Attendance ID', 'Student Name', 'Batch', 'Timestamp'])
        
        # Write CSV rows
        for record in attendance_records:
            writer.writerow([record['attendance_id'], f"{record['first_name']} {record['last_name']}", record['batch'], record['timestamp']])
        
        # Generate the response
        output.seek(0)  # Reset the pointer to the start of the file
        return Response(output, mimetype='text/csv', headers={
            "Content-Disposition": "attachment;filename=attendance_sheet.csv"
        })
    

    conn.close()

    # Render the template with attendance data
    return render_template('admin_attendance.html', attendance_records=attendance_records, search_query=search_query)


@app.route("/logout")
def logout():
    session.clear()  # Clear session data
    return redirect(url_for("login"))

if __name__ == "__main__":
    initialize_database()  # Ensure tables exist
    app.run(debug=True)
