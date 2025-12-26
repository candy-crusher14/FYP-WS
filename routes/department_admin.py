from flask import Blueprint, render_template, session, redirect, url_for, flash
from extensions import db
from models import Admin, Department, Course, Submission, Assignment, Classroom
from sqlalchemy import func

department_admin_bp = Blueprint('department_admin', __name__, url_prefix='/department_admin')

@department_admin_bp.route('/dashboard')
def dashboard():
    if 'admin_logged_in' not in session or session.get('role') != 'department_head':
        flash('Access denied. You must be a department head to view this page.', 'danger')
        return redirect(url_for('login'))

    admin_id = session['admin_id']
    # Find the department this admin is the head of
    department = Department.query.filter_by(head_id=admin_id).first_or_404()

    # Aggregate performance data
    courses = Course.query.filter_by(department_id=department.id).all()
    course_performance = []

    for course in courses:
        # Basic aggregation logic, can be expanded
        avg_score = db.session.query(func.avg(Submission.score)) \
            .join(Assignment) \
            .join(Classroom) \
            .filter(Classroom.course_id == course.id).scalar() or 0.0

        total_submissions = db.session.query(func.count(Submission.id)) \
            .join(Assignment) \
            .join(Classroom) \
            .filter(Classroom.course_id == course.id).scalar() or 0

        failing_submissions = db.session.query(func.count(Submission.id)) \
            .join(Assignment) \
            .join(Classroom) \
            .filter(Classroom.course_id == course.id, Submission.score < 50).scalar() or 0 # Assuming 50 is the failing threshold
        
        failing_rate = (failing_submissions / total_submissions) * 100 if total_submissions > 0 else 0

        course_performance.append({
            'course_code': course.course_code,
            'title': course.title,
            'average_gpa': (avg_score / 100) * 4.0, # Simple conversion to 4.0 scale
            'failing_rate': round(failing_rate, 2)
        })

    return render_template('department_dashboard.html', department=department, performance_data=course_performance)
