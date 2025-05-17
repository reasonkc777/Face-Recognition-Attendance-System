## Facial Recognition Attendance System

A Flask-based web application for managing and recording student attendance using face recognition powered by FaceNet.
For accessing this project head in next branch named Code.

> **This project was developed as a Final Year Project by a 4th-year CSIT student.**

---

### Features

* Face Detection with OpenCV Haar cascades
* Face Embeddings via Keras FaceNet
* Student Registration with face data capture
* Automated Attendance marking via webcam
* Admin Dashboard for managing students and attendance
* Student Dashboard to view personal attendance
* CSV Export of attendance records
* Role-Based Login (Admin/Student)

---

### Project Structure

```
attendance-system/
├── user_data/                 # Captured face images per student
├── templates/                # HTML templates
├── static/                   # CSS, JS, assets 
├── attendance.db             # SQLite database
├── app.py                    # Main Flask app
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

### Prerequisites

Ensure you have Python 3.7 or higher installed.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

### Dependencies

Add the following to your `requirements.txt`:

```
Flask  
opencv-python  
keras-facenet  
numpy  
pillow  
```

Note: `sqlite3` is part of the Python standard library and does not need to be installed separately.

---

### Running the App

1. Start the Flask server:

```bash
python app.py
```

2. Open your browser and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Student Workflow

1. Register on `/register`
2. Capture face data on `/capture_student`
3. Log in and mark attendance using the webcam

---

### Admin Workflow

1. Log in via `/login`
2. Use the admin dashboard:

   * `/admin_dashboard`
   * `/manage_students`
   * `/attendance`
3. Export attendance as a CSV file

---

### Database Reset

To clear all records but keep the table structure, run these SQL commands:

```sql
DELETE FROM students;
DELETE FROM student_embedding;
DELETE FROM attendance;
DELETE FROM admin;
```

Alternatively, delete the database file:

```bash
rm attendance.db
```

Then restart the app. The database and tables will be recreated automatically.