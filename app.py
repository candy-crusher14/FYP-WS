from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import sys
import string
import warnings
from sqlalchemy import func, and_, or_, text, event
from sqlalchemy.exc import OperationalError, SQLAlchemyError, IntegrityError, DataError
import google.generativeai as genai
import PIL.Image
from collections import defaultdict
import os
import hashlib
import re
import random
import io
import uuid  # For invite codes
from datetime import datetime, timedelta
import html  # For HTML escaping
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import json
from dotenv import load_dotenv
from forms import TeacherRegistrationForm
from flask_wtf.csrf import CSRFProtect

load_dotenv()

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_dev_123!@#')
csrf = CSRFProtect(app)

# File upload configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx'}

# Create upload directories
import os
for folder in ['uploads/profiles', 'uploads/documents']:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    
    # Check file extension
    extension = filename.rsplit('.', 1)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    
    # Additional security checks
    # Prevent double extensions like .php.jpg
    if filename.count('.') > 1:
        # Allow only if the second-to-last part is not executable
        parts = filename.split('.')
        if len(parts) > 2 and parts[-2].lower() in {'php', 'jsp', 'asp', 'aspx', 'exe', 'bat', 'cmd', 'sh'}:
            return False
    
    # Prevent null bytes and other dangerous characters
    if '\x00' in filename or any(char in filename for char in ['<', '>', ':', '"', '|', '?', '*']):
        return False
    
    return True

# Removed the problematic session clearing block that was here
print('changes')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///exam_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True, 'pool_recycle': 300}
app.config['UPLOAD_FOLDER'] = 'uploads'

from extensions import db
db.init_app(app)

gemini_model = None
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file. AI features will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        print("Gemini AI model initialized successfully with 'models/gemini-2.5-flash'.")
    except Exception as e:
        print(f"Error initializing Gemini AI model: {e}")

DEPARTMENTS = ('Computer Science', 'Information Technology')
SEMESTERS_LIST = [f"Semester {i}" for i in range(1, 9)]
UNIVERSITIES = ('University of Mirpur Khas', 'University of Sindh')

# Helper Functions

def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()

def sanitize_input(text):
    """Sanitize user input to prevent XSS attacks"""
    if not text:
        return text
    # HTML escape the input
    sanitized = html.escape(str(text).strip())
    # Remove any remaining script tags or javascript
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    return sanitized


def _sanitize_gemini_response(raw_text):
    """Strips Markdown formatting from a raw Gemini response."""
    # Find the start and end of the JSON block
    json_start = raw_text.find('{')
    json_end = raw_text.rfind('}')
    if json_start != -1 and json_end != -1:
        return raw_text[json_start:json_end+1]
    return raw_text # Return as-is if no JSON block is found


def _generate_and_save_clusters(assignment_id):
    """Helper function to generate and save clusters for a given assignment."""
    if not gemini_model:
        return "Error: Gemini model not configured."

    submissions = Submission.query.filter(
        Submission.assignment_id == assignment_id,
        Submission.answer_text != None,
        Submission.answer_text != ''
    ).all()

    if len(submissions) < 2:
        return f"Skipped: Less than 2 submissions, no clustering needed."

    anonymous_data = {}
    id_map = {}
    for i, s in enumerate(submissions):
        anon_id = f"response_{i+1}"
        anonymous_data[anon_id] = s.answer_text
        id_map[anon_id] = s.id

    prompt = f"""
    You are an expert pedagogical assistant. Group student responses based on conceptual understanding.
    Student Responses: {json.dumps(anonymous_data, indent=2)}
    Return a single JSON object with a key 'groups'. Each group must have:
    1. 'submission_ids': A list of the anonymous string IDs (e.g., ["response_1", "response_5"]).
    2. 'conceptual_summary': A string explaining the group's shared concept.
    """

    response = gemini_model.generate_content(prompt)
    raw_text = _sanitize_gemini_response(response.text)
    parsed_json = json.loads(raw_text)
    grouped_results = parsed_json if isinstance(parsed_json, dict) else {"groups": parsed_json}

    with db.session.begin_nested():
        for i, group in enumerate(grouped_results.get('groups', [])):
            new_cluster = Cluster(
                assignment_id=assignment_id,
                name=f"Cluster {i + 1}",
                conceptual_summary=group.get('conceptual_summary', 'No summary provided.')
            )
            db.session.add(new_cluster)
            db.session.flush()

            real_submission_ids = [id_map[anon_id] for anon_id in group.get('submission_ids', []) if anon_id in id_map]

            for sub_id in real_submission_ids:
                cs = ClusteredSubmission(cluster_id=new_cluster.id, submission_id=sub_id)
                db.session.add(cs)
    db.session.commit()
    return "Success"


@app.route('/teacher/assignment/<int:assignment_id>/cluster', methods=['POST'])
def cluster_assignment(assignment_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login', role='teacher'))
    
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=session['teacher_id']).first_or_404()
    
    try:
        # For now, we assume this is the first clustering. Re-clustering logic will be added later.
        existing_clusters = Cluster.query.filter_by(assignment_id=assignment.id).count()
        if existing_clusters > 0:
            flash('This assignment has already been clustered.', 'info')
        else:
            result_message = _generate_and_save_clusters(assignment_id)
            if result_message and 'Skipped' in result_message:
                flash(f'Clustering skipped: Not enough submissions for this assignment.', 'warning')
            elif result_message and 'Error' in result_message:
                flash(f'An error occurred during clustering: {result_message}', 'danger')
            else:
                flash('Successfully generated AI clusters for the assignment!', 'success')
    except Exception as e:
        flash(f'An error occurred during clustering: {str(e)}', 'danger')

    return redirect(url_for('teacher_dashboard', active_tab='evaluation-hub', filter_assignment_id=assignment_id))


@app.route('/teacher/assignment/<int:assignment_id>/recluster', methods=['POST'])
def recluster_assignment(assignment_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login', role='teacher'))

    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=session['teacher_id']).first_or_404()

    try:
        # Delete existing clusters for this assignment
        Cluster.query.filter_by(assignment_id=assignment.id).delete()
        db.session.commit()

        result_message = _generate_and_save_clusters(assignment_id)
        if result_message and 'Skipped' in result_message:
            flash(f'Re-clustering skipped: Not enough submissions for this assignment.', 'warning')
        elif result_message and 'Error' in result_message:
            flash(f'An error occurred during re-clustering: {result_message}', 'danger')
        else:
            flash('Successfully re-generated AI clusters for the assignment!', 'success')
    except Exception as e:
        flash(f'An error occurred during re-clustering: {str(e)}', 'danger')

    return redirect(url_for('teacher_dashboard', active_tab='evaluation-hub', filter_assignment_id=assignment_id))


# Routes
@app.route('/')
def index():
    if 'admin_logged_in' in session: return redirect(url_for('admin_dashboard'))
    if 'teacher_logged_in' in session: return redirect(url_for('teacher_dashboard'))
    if 'student_logged_in' in session: return redirect(url_for('student_dashboard'))
    return render_template('homepage.html')


@app.route('/dashboard')
def dashboard():
    if 'admin_logged_in' in session:
        return redirect(url_for('admin_dashboard'))
    elif 'teacher_logged_in' in session:
        return redirect(url_for('teacher_dashboard'))
    elif 'student_logged_in' in session:
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))


# Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # Don't hash here, compare with stored hash
        admin_code = request.form.get('admin_code', '')
        if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
            flash('Invalid admin code', 'danger')
            return render_template('login.html', error='Invalid admin code', default_role='admin')
        
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.password == hash_password(password):
            session['admin_logged_in'], session['admin_id'] = True, admin.id
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials', 'danger')
    return render_template('login.html', default_role='admin')


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('Please log in as admin.', 'warning');
        return redirect(url_for('login', role='admin'))
    
    admin = Admin.query.get_or_404(session['admin_id'])
    
    stats = {
        'students': Student.query.count(), 'teachers': Teacher.query.count(),
        'classrooms': Classroom.query.count(), 'assignments': Assignment.query.count(),
        'recent_assignments': Assignment.query.order_by(Assignment.created_at.desc()).limit(5).all()
    }
    pending_teachers = Teacher.query.filter(Teacher.is_approved == False).order_by(Teacher.created_at.desc()).all()

    return render_template('admin_dashboard.html', stats=stats, pending_teachers=pending_teachers, admin=admin)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None);
    session.pop('admin_id', None)
    flash('Admin logged out successfully.', 'info');
    return redirect(url_for('index'))


# Teacher Routes
@app.route('/teacher/register', methods=['GET', 'POST'])
def teacher_register():
    form = TeacherRegistrationForm()
    form.university_id.choices = [(u.id, u.name) for u in University.query.order_by('name').all()]

    if form.validate_on_submit():
        hashed_password = hash_password(form.password.data)
        new_teacher = Teacher(
            username=form.username.data,
            email=form.email.data,
            password=hashed_password,
            full_name=form.full_name.data,
            university_id=form.university_id.data,
            is_approved=False
        )
        db.session.add(new_teacher)
        db.session.commit()
        flash(f'Hi {form.full_name.data}! Your teacher account is pending admin approval.', 'info')
        return redirect(url_for('index'))

    return render_template('teacher_register.html', form=form)


@app.route('/teacher/login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        teacher = Teacher.query.filter_by(username=username).first()
        if teacher and teacher.password == hash_password(password):
            if not teacher.is_approved:
                flash(f'Your account is pending approval from {teacher.university.name} admin. Please wait for verification.', 'warning')
                return render_template('login.html', default_role='teacher')
            session['teacher_logged_in'], session['teacher_id'] = True, teacher.id
            flash('Teacher login successful!', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Invalid teacher credentials.', 'danger')
    return render_template('login.html', default_role='teacher')


@app.route('/teacher/dashboard')
def teacher_dashboard():
    if not session.get('teacher_logged_in'):
        flash('Please log in as a teacher.', 'warning')
        return redirect(url_for('login', role='teacher'))
    try:
        teacher_id = session['teacher_id']
        teacher = Teacher.query.get_or_404(teacher_id)

        filter_classroom_id = request.args.get('filter_classroom_id', type=int)
        filter_assignment_id = request.args.get('filter_assignment_id', type=int)
        active_tab_on_load = request.args.get('active_tab', 'teacher-dashboard')

        classrooms = teacher.classrooms.order_by(Classroom.name).all()
        all_assignments_for_filter_dropdown = teacher.assignments.order_by(Assignment.title).all()

        assignments_base_query = teacher.assignments
        if filter_classroom_id:
            assignments_base_query = assignments_base_query.filter(Assignment.classroom_id == filter_classroom_id)

        report_assignments = assignments_base_query.order_by(Assignment.created_at.desc()).all()
        if filter_assignment_id:
            report_assignments = [a for a in report_assignments if a.id == filter_assignment_id]

        active_assignments_list_query = teacher.assignments.filter(Assignment.submission_deadline > datetime.utcnow())
        if filter_classroom_id and active_tab_on_load == 'teacher-dashboard':
            active_assignments_list_query = active_assignments_list_query.filter(
                Assignment.classroom_id == filter_classroom_id)
        active_assignments_list = active_assignments_list_query.order_by(Assignment.submission_deadline.asc()).all()

        total_enrollments = sum(c.enrollments_count for c in classrooms)
        assignment_ids_by_teacher = [a.id for a in teacher.assignments.all()]

        pending_submissions_query = Submission.query.filter(
            Submission.assignment_id.in_(assignment_ids_by_teacher),
            Submission.submitted_at != None, Submission.evaluated_at == None
        )
        completed_submissions_query = Submission.query.filter(
            Submission.assignment_id.in_(assignment_ids_by_teacher),
            Submission.evaluated_at != None
        )

        if filter_classroom_id:
            assignments_in_filtered_classroom_ids = [a.id for a in
                                                     Assignment.query.filter_by(classroom_id=filter_classroom_id,
                                                                                teacher_id=teacher_id).all()]
            pending_submissions_query = pending_submissions_query.filter(
                Submission.assignment_id.in_(assignments_in_filtered_classroom_ids))
            completed_submissions_query = completed_submissions_query.filter(
                Submission.assignment_id.in_(assignments_in_filtered_classroom_ids))

        if filter_assignment_id:
            pending_submissions_query = pending_submissions_query.filter(Submission.assignment_id == filter_assignment_id)
            completed_submissions_query = completed_submissions_query.filter(
                Submission.assignment_id == filter_assignment_id)

        pending_submissions = pending_submissions_query.order_by(Submission.submitted_at.desc()).all()
        completed_submissions = completed_submissions_query.order_by(Submission.evaluated_at.desc()).all()
        
        classroom_ids = [c.id for c in classrooms]
        pending_enrollments = Enrollment.query.filter(
            Enrollment.classroom_id.in_(classroom_ids),
            Enrollment.is_approved == False
        ).join(Student).join(Classroom).order_by(Enrollment.enrolled_at.desc()).all()

        return render_template('teacher_dashboard.html',
                               teacher=teacher, classrooms=classrooms,
                               assignments=report_assignments,
                               all_assignments_for_filter=all_assignments_for_filter_dropdown,
                               active_assignments=active_assignments_list,
                               total_enrollments=total_enrollments,
                               pending_evaluations=len(pending_submissions),
                               pending_submissions=pending_submissions,
                               completed_submissions=completed_submissions,
                               pending_enrollments=pending_enrollments,
                               current_filter_classroom_id=filter_classroom_id,
                               current_filter_assignment_id=filter_assignment_id,
                               active_tab_on_load=active_tab_on_load)
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'danger')
        return redirect(url_for('teacher_dashboard'))

def create_assignment():
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login', role='teacher'))
    try:
        teacher_id = session['teacher_id']
        title = request.form.get('title', '').strip()
        classroom_id_str = request.form.get('classroom_id')
        instructions = request.form.get('instructions', '').strip()
        deadline_days_str = request.form.get('deadline_days')
        evaluation_method = request.form.get('evaluation_method')
        expected_answer = request.form.get('expected_answer', '').strip()
        max_score_str = request.form.get('max_score')

        if not all([title, classroom_id_str, instructions, deadline_days_str, evaluation_method, max_score_str]):
            flash('All assignment fields are required.', 'danger')
        else:
            classroom_id = int(classroom_id_str)
            deadline_days = int(deadline_days_str)
            max_score = float(max_score_str)

            if deadline_days <= 0:
                flash('Deadline must be at least 1 day.', 'danger')
            elif max_score <= 0:
                flash('Max score must be positive.', 'danger')
            else:
                classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=teacher_id).first_or_404()
                
                if evaluation_method == 'expected' and not expected_answer:
                    flash('Expected answer is required for that evaluation method.', 'danger')
                else:
                    submission_deadline = datetime.utcnow() + timedelta(days=deadline_days)
                    new_assignment = Assignment(
                        title=title, classroom_id=classroom_id, teacher_id=teacher_id, 
                        instructions=instructions, submission_deadline=submission_deadline, 
                        evaluation_method=evaluation_method, 
                        expected_answer=expected_answer if evaluation_method == 'expected' else None,
                        max_score=max_score
                    )
                    db.session.add(new_assignment)
                    db.session.commit()
                    flash(f'Assignment "{title}" created successfully!', 'success')
    except (ValueError, TypeError):
        flash('Invalid data format for form fields.', 'danger')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Database error: {str(e)}', 'danger')
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'danger')
    finally:
        return redirect(url_for('teacher_dashboard', active_tab='create-assignment'))


@app.route('/teacher/submission/<int:submission_id>/override_score', methods=['POST'])
def override_score(submission_id):
    if not session.get('teacher_logged_in'):
        flash('Authentication required.', 'danger')
        return redirect(url_for('login', role='teacher'))
    
    try:
        submission = Submission.query.get_or_404(submission_id)
        assignment = Assignment.query.get_or_404(submission.assignment_id)

        if assignment.teacher_id != session['teacher_id']:
            flash('You do not have permission to override this score.', 'danger')
        else:
            new_score_str = request.form.get('new_score')
            reason = request.form.get('reason', '').strip()
            if not new_score_str or not reason:
                flash('New score and reason are required.', 'danger')
            else:
                new_score = float(new_score_str)
                if not (0 <= new_score <= assignment.max_score):
                    flash(f'Score must be between 0 and {assignment.max_score}.', 'danger')
                else:
                    submission.override_score = new_score
                    submission.override_reason = reason
                    submission.overridden_at = datetime.utcnow()
                    submission.overridden_by = session['teacher_id']
                    submission.evaluated_at = datetime.utcnow()
                    db.session.commit()
                    flash('Score override successful!', 'success')
    except ValueError:
        flash('Invalid score format. Please enter a number.', 'danger')
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Database error during override: {str(e)}', 'danger')
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'danger')
    finally:
        # Preserve filters on redirect
        redirect_url = url_for('teacher_dashboard', active_tab='evaluation-hub')
        if request.args.get('filter_assignment_id'):
            redirect_url = url_for('teacher_dashboard', active_tab='evaluation-hub', filter_assignment_id=request.args.get('filter_assignment_id'))
        return redirect(redirect_url)


@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        student = Student.query.filter_by(username=username).first()
        if student and student.password == hash_password(password):
            session['student_logged_in'], session['student_id'] = True, student.id
            flash('Student login successful!', 'success');
def student_dashboard():
    if not session.get('student_logged_in'):
        flash('Please log in as a student.', 'warning')
        return redirect(url_for('login', role='student'))
    try:
        student_id = session['student_id']
        student = Student.query.get_or_404(student_id)

        enrolled_classrooms_ids = [e.classroom_id for e in student.enrollments.filter(Enrollment.is_approved == True).all()]
        enrolled_classrooms = Classroom.query.filter(Classroom.id.in_(enrolled_classrooms_ids)).all() if enrolled_classrooms_ids else []
        
        submitted_as_ids = db.session.query(Submission.assignment_id).filter(
            Submission.student_id == student.id,
            Submission.submitted_at != None
        )
        
        available_assignments = []
        if enrolled_classrooms_ids:
            available_assignments = Assignment.query.filter(
                Assignment.classroom_id.in_(enrolled_classrooms_ids),
                Assignment.submission_deadline > datetime.utcnow(),
                ~Assignment.id.in_(submitted_as_ids)
            ).order_by(Assignment.submission_deadline.asc()).all()
        
        recent_submissions = Submission.query.filter_by(student_id=student.id) \
            .filter(Submission.submitted_at != None) \
            .join(Assignment).order_by(Submission.submitted_at.desc()).limit(5).all()

        total_submissions = student.submissions.filter(Submission.submitted_at != None).count()
        evaluated_submissions = student.submissions.filter(Submission.evaluated_at != None).count()
        
        evaluated_subs = student.submissions.filter(
            Submission.evaluated_at != None,
            Submission.display_score != None
        ).all()
        
        avg_score = 0
        if evaluated_subs:
            total_score = sum(sub.display_score or 0 for sub in evaluated_subs)
            total_max = sum(sub.assignment.max_score or 0 for sub in evaluated_subs)
            avg_score = (total_score / total_max * 100) if total_max > 0 else 0
        
        return render_template('student_dashboard.html', 
                             student=student, 
                             available_assignments=available_assignments,
                             recent_submissions=recent_submissions, 
                             enrolled_classrooms=enrolled_classrooms,
                             total_submissions=total_submissions,
                             evaluated_submissions=evaluated_submissions,
                             avg_score=avg_score)
    except Exception as e:
        flash(f'An error occurred loading your dashboard: {str(e)}', 'danger')
        return redirect(url_for('login', role='student'))


@app.route('/student/join_classroom', methods=['GET', 'POST'])
def student_join_classroom():
    if not session.get('student_logged_in'):
        flash('Please log in to join a classroom.', 'warning')
        return redirect(url_for('login', role='student'))
    student = Student.query.get_or_404(session['student_id'])

    if request.method == 'POST':
        try:
            invite_code = request.form.get('invite_code', '').strip().upper()
            if not invite_code:
                flash('Invite code is required.', 'danger')
            else:
                classroom = Classroom.query.filter_by(invite_code=invite_code).first()
                if not classroom:
                    flash('Invalid invite code. Classroom not found.', 'danger')
                elif classroom.year != student.year or classroom.department != student.department:
                    flash(f'You cannot join this classroom. It is for Year {classroom.year} {classroom.department} students.', 'danger')
                else:
                    existing_enrollment = Enrollment.query.filter_by(student_id=student.id, classroom_id=classroom.id).first()
                    if existing_enrollment:
                        flash(f'You are already enrolled or pending approval for "{classroom.name}".', 'info')
                    else:
                        enrollment = Enrollment(student_id=student.id, classroom_id=classroom.id, is_approved=False)
                        db.session.add(enrollment)
                        db.session.commit()
                        flash(f'Join request sent for "{classroom.name}"!', 'success')
                        return redirect(url_for('student_dashboard'))
        except SQLAlchemyError as e:
            db.session.rollback()
            flash(f'Database error: {str(e)}', 'danger')
        except Exception as e:
            flash(f'An unexpected error occurred: {str(e)}', 'danger')
            
    return render_template('student_join_classroom.html', student=student)
    if not session.get('student_logged_in'):
        flash('Please log in to submit.', 'warning')
        return redirect(url_for('login', role='student'))
    try:
        student_id = session['student_id']
        assignment = Assignment.query.get_or_404(assignment_id)
        student = Student.query.get_or_404(student_id)

        enrollment = Enrollment.query.filter_by(student_id=student_id, classroom_id=assignment.classroom_id, is_approved=True).first()
        if not enrollment:
            flash('You are not enrolled in this assignment\'s classroom.', 'danger')
            return redirect(url_for('student_dashboard'))

        existing_submission = Submission.query.filter_by(student_id=student_id, assignment_id=assignment_id).first()
        if existing_submission and existing_submission.submitted_at:
            flash('You have already submitted this assignment.', 'info')
            return redirect(url_for('student_dashboard'))

        if assignment.submission_deadline < datetime.utcnow():
            flash('The deadline for this assignment has passed.', 'danger')
            return redirect(url_for('student_dashboard'))

        if request.method == 'POST':
            answer_text = request.form.get('answer_text', '').strip()
            if not answer_text:
                flash('Answer text cannot be empty.', 'danger')
            else:
                target_submission = existing_submission or Submission(assignment_id=assignment_id, student_id=student_id)
                target_submission.answer_text = answer_text
                target_submission.submitted_at = datetime.utcnow()
                if not existing_submission: 
                    db.session.add(target_submission)
                db.session.commit()
                flash('Assignment submitted successfully!', 'success')
                return redirect(url_for('student_dashboard'))
    except SQLAlchemyError as e:
        db.session.rollback()
        flash(f'Database error: {str(e)}', 'danger')
        return redirect(url_for('student_dashboard'))
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'danger')
        return redirect(url_for('student_dashboard'))

    return render_template('submit_assignment.html', assignment=assignment, student=student)


@app.route('/student/assignment/<int:assignment_id>/history')
def student_submission_history(assignment_id):
    if not session.get('student_logged_in'):
        flash('Please log in to view your submission history.', 'warning')
        return redirect(url_for('login', role='student'))

    student_id = session['student_id']
    assignment = Assignment.query.get_or_404(assignment_id)
    submission = Submission.query.filter_by(student_id=student_id, assignment_id=assignment_id).first()

    if not submission:
        flash('You have not made a submission for this assignment.', 'info')
        return redirect(url_for('student_dashboard'))

    return render_template('student_submission_history.html', assignment=assignment, submission=submission)


@app.route('/teacher/report/assignment/<int:assignment_id>')
def teacher_report_assignment(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    teacher_id = session['teacher_id']
    assignment = Assignment.query.filter_by(id=assignment_id, teacher_id=teacher_id).first_or_404()
    teacher = Teacher.query.get_or_404(teacher_id)
    submissions_with_students = db.session.query(Submission, Student) \
        .join(Student, Submission.student_id == Student.id) \
        .filter(Submission.assignment_id == assignment_id) \
        .order_by(Student.full_name).all()

    return render_template('teacher_report_assignment.html',
                           assignment=assignment, teacher=teacher,
                           submissions=submissions_with_students)


# Analytics Routes
@app.route('/teacher/analytics')
def teacher_analytics():
    if not session.get('teacher_logged_in'):
        flash('Please log in as a teacher.', 'warning')
        return redirect(url_for('login', role='teacher'))
    
    teacher_id = session['teacher_id']
    teacher = Teacher.query.get_or_404(teacher_id)
    
    # Get analytics data
    total_assignments = teacher.assignments.count()
    total_classrooms = teacher.classrooms.count()
    
    assignment_ids = [a.id for a in teacher.assignments]
    
    total_submissions = Submission.query.filter(Submission.assignment_id.in_(assignment_ids)).count()
    evaluated_submissions = Submission.query.filter(Submission.assignment_id.in_(assignment_ids), Submission.evaluated_at != None).count()
    
    # Get evaluation method breakdown
    ai_evaluations = Submission.query.join(Assignment).filter(
        Assignment.teacher_id == teacher_id,
        Submission.evaluation_method == 'gemini'
    ).count()
    
    manual_evaluations = Submission.query.join(Assignment).filter(
        Assignment.teacher_id == teacher_id,
        Submission.evaluation_method == 'expected'
    ).count()
    
    # Get average scores
    avg_score_query = db.session.query(func.avg(Submission.score)).join(Assignment).filter(
        Assignment.teacher_id == teacher_id,
        Submission.evaluated_at != None
    ).scalar()
    avg_score = round(avg_score_query, 2) if avg_score_query else 0
    
    analytics_data = {
        'total_assignments': total_assignments,
        'total_classrooms': total_classrooms,
        'total_submissions': total_submissions,
        'evaluated_submissions': evaluated_submissions,
        'pending_evaluations': total_submissions - evaluated_submissions,
        'ai_evaluations': ai_evaluations,
        'manual_evaluations': manual_evaluations,
        'avg_score': avg_score,
        'evaluation_rate': round((evaluated_submissions / total_submissions * 100), 2) if total_submissions > 0 else 0
    }
    
    return render_template('teacher_analytics.html', teacher=teacher, analytics=analytics_data)


# Combined Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'admin_logged_in' in session: return redirect(url_for('admin_dashboard'))
    if 'teacher_logged_in' in session: return redirect(url_for('teacher_dashboard'))
    if 'student_logged_in' in session: return redirect(url_for('student_dashboard'))
    default_role = request.args.get('role', 'student')
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form.get('role', 'student')
        admin_code = request.form.get('admin_code', '')
        if not username or not password:
            flash('Username and password are required.', 'danger')
        else:
            user = None
            if role == 'teacher':
                user = Teacher.query.filter_by(username=username).first()
                if user and user.password != hash_password(password):
                    user = None
            elif role == 'student':
                user = Student.query.filter_by(username=username).first()
                if user and user.password != hash_password(password):
                    user = None
            elif role == 'admin':
                if admin_code != os.environ.get('ADMIN_SECRET', 'admin123'):
                    flash('Invalid admin code.', 'danger')
                else:
                    user = Admin.query.filter_by(username=username).first()
                    if user and user.password != hash_password(password):
                        user = None
            if user:
                # Check teacher approval status
                if role == 'teacher' and not user.is_approved:
                    flash(f'Your account is pending approval from {user.university} admin. Please wait for verification.', 'warning')
                    return render_template('login.html', default_role=role, submitted_username=username)
                
                session[f'{role}_logged_in'], session[f'{role}_id'] = True, user.id
                flash('Login successful!', 'success');
                return redirect(url_for(f'{role}_dashboard'))
            else:
                flash(f'Invalid credentials or role for {role}.', 'danger')
        return render_template('login.html', error='Login failed.', default_role=role, submitted_username=username)
    return render_template('login.html', default_role=default_role)


# Placeholder routes that now flash messages
@app.route('/teacher/report/classroom/<int:classroom_id>')
def classroom_report(classroom_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    classroom = Classroom.query.get_or_404(classroom_id)
    if classroom.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Classroom-specific PDF report for '{classroom.name}' is under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='classroom-manager'))


@app.route('/teacher/submissions/assignment/<int:assignment_id>')
def view_submissions(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    assignment = Assignment.query.get_or_404(assignment_id)
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Displaying submissions for '{assignment.title}' in Evaluation Hub.", "info")
    return redirect(url_for('teacher_dashboard', filter_assignment_id=assignment_id, active_tab='evaluation-hub'))


@app.route('/teacher/submission/<int:submission_id>/view_details')
def teacher_view_submission_details(submission_id):
    if not session.get('teacher_logged_in'): 
        return redirect(url_for('login', role='teacher'))
    
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get_or_404(submission.assignment_id)
    
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger")
        return redirect(url_for('teacher_dashboard'))
    
    # Get student information
    student = submission.student
    
    # Get other submissions by this student for comparison
    other_submissions = Submission.query.filter(
        Submission.student_id == student.id,
        Submission.id != submission.id,
        Submission.submitted_at != None
    ).join(Assignment).order_by(Submission.submitted_at.desc()).limit(5).all()
    
    return render_template('teacher_submission_details.html',
                         submission=submission,
                         assignment=assignment,
                         student=student,
                         other_submissions=other_submissions)


@app.route('/teacher/assignment/<int:assignment_id>/analytics')
def assignment_analytics(assignment_id):
    if not session.get('teacher_logged_in'): return redirect(url_for('login', role='teacher'))
    assignment = Assignment.query.get_or_404(assignment_id)
    if assignment.teacher_id != session.get('teacher_id'):
        flash("Permission denied.", "danger");
        return redirect(url_for('teacher_dashboard'))
    flash(f"Analytics for assignment '{assignment.title}' is under development.", "info")
    return redirect(url_for('teacher_dashboard', active_tab='reports'))


# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    # Don't flash error messages, just redirect silently
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


@app.errorhandler(SQLAlchemyError)
def handle_db_error(e):
    error_type = type(e).__name__;
    error_details = str(e)
    orig_error = getattr(e, 'orig', None)
    if orig_error: error_details += f" (Original error: {str(orig_error)})"
    full_error_message = f"Database Error ({error_type}): {error_details}"
    print(full_error_message, file=sys.stderr);
    app.logger.error(full_error_message)
    db.session.rollback()
    user_message = "A database operation failed."
    if isinstance(e, IntegrityError):
        user_message = "Failed to save: data conflicts with existing records (e.g., duplicate username/email or invalid reference). Check input."
    elif isinstance(e, DataError):
        user_message = "Failed to save: incorrect data format for the database."
    elif isinstance(e, OperationalError):
        user_message = "Database connection/operational error. Try again or contact support."
    flash_message = f'{user_message} (Details logged for admin)' if not app.debug else f'{user_message} Details: {full_error_message}'
    flash(flash_message, 'danger')
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


@app.errorhandler(500)
def internal_server_error(e):
    print(f"FATAL: Internal Server Error: {e}", file=sys.stderr)
    app.logger.error(f"Internal Server Error: {e}", exc_info=True)
    flash('Error 500: An unexpected internal server error occurred. Please try again or contact support.', 'danger')
    target_dashboard = 'index'
    if 'teacher_logged_in' in session:
        target_dashboard = 'teacher_dashboard'
    elif 'student_logged_in' in session:
        target_dashboard = 'student_dashboard'
    elif 'admin_logged_in' in session:
        target_dashboard = 'admin_dashboard'
    return redirect(url_for(target_dashboard))


# Register Blueprints
from routes.department_admin import department_admin_bp
from seed import seed_test_data

app.register_blueprint(department_admin_bp)
app.cli.add_command(seed_test_data)

# Main Execution
@app.route('/dev/test-cluster/<int:assignment_id>')
def test_cluster_route(assignment_id):
    """A developer route to test the answer grouping feature."""
    if not app.debug:
        return jsonify({'error': 'This route is only available in debug mode.'}), 403

    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
    if not submissions:
        all_ids = [a.id for a in Assignment.query.with_entities(Assignment.id).all()]
        return jsonify({
            'error': f'No submissions found for assignment ID {assignment_id}.',
            'available_assignment_ids': all_ids
        }), 404

    # Trigger the clustering
    raw_ai_response = cluster_submissions(assignment_id)
    
    # Fetch the results
    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
    
    # Group students by the new group_id
    grouped_students = defaultdict(list)
    for sub in submissions:
        if sub.group_id:
            grouped_students[sub.group_id].append(sub.student.username)

    # Format the response
    response_data = {f'Group {gid}': students for gid, students in grouped_students.items()}

    return jsonify({
        'assignment_id': assignment_id,
        'status': 'Clustering complete.',
        'groups': response_data,
        'raw_ai_response': raw_ai_response
    })


if __name__ == '__main__':
    from models import (Admin, Student, Teacher, Classroom, Assignment, Submission, 
                        Enrollment, StudentDocument, TeacherDocument, University, 
                        Department, Course, Rubric, RubricItem, AppliedRubricItem)
    print("Reminder: If models changed & using SQLite, delete 'exam_system.db' for dev schema reset (ERASES DATA).")
    with app.app_context():
        try:
            db.create_all();
            print("DB tables created/ensured.")
            # Verification Query: Print all assignment IDs
            all_assignment_ids = [a.id for a in Assignment.query.with_entities(Assignment.id).all()]
            print(f"[INFO] Existing Assignment IDs in DB: {all_assignment_ids}")
            if not Admin.query.first():
                # Create default system admin
                db.session.add(
                    Admin(username='admin', password=hash_password(os.environ.get('DEFAULT_ADMIN_PASS', 'admin123')),
                          full_name='System Administrator', role='admin'))
                print(f"Default admin 'admin' created.")
                
                # This section is commented out as it requires a UI to properly associate admins with universities.
                pass
                print(f"University admin for 'University of Sindh' created.")
            if not Teacher.query.first():
                teacher = Teacher(username='teacher',
                                  password=hash_password(os.environ.get('DEFAULT_TEACHER_PASS', 'teacher123')),
                                  full_name='Ada Lovelace', email='ada.lovelace@example.com',
                                  university='University of Mirpur Khas', is_approved=True)
                db.session.add(teacher);
                db.session.commit()
                print(f"Default teacher 'teacher' created.")
                if not Classroom.query.first():
                    classroom = Classroom(name='Intro to CS', teacher_id=teacher.id, department='Computer Science',
                                          year=1, description='Fundamentals.')
                    db.session.add(classroom)
                    db.session.commit()
                    print(
                        f"Default classroom '{classroom.name}' created for teacher '{teacher.username}'. Invite: {classroom.invite_code}")
            db.session.commit()
        except OperationalError as oe:
            print(f"CRITICAL DB ERROR: {oe}", file=sys.stderr); sys.exit(1)
        except Exception as e:
            print(f"Setup Error: {e}", file=sys.stderr); db.session.rollback()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))