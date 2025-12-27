from extensions import db
from sqlalchemy import func, event, text
from sqlalchemy.orm.attributes import get_history
from datetime import datetime
import uuid

# --- Core Educational Structure Models ---

class University(db.Model):
    __tablename__ = 'universities'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    departments = db.relationship('Department', backref='university', lazy='dynamic', cascade="all, delete-orphan")

class Department(db.Model):
    __tablename__ = 'departments'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    university_id = db.Column(db.Integer, db.ForeignKey('universities.id'), nullable=False)
    courses = db.relationship('Course', backref='department', lazy='dynamic', cascade="all, delete-orphan")
    head_id = db.Column(db.Integer, db.ForeignKey('admins.id'), nullable=True)
    __table_args__ = (db.UniqueConstraint('name', 'university_id', name='_dept_uni_uc'),)

class Course(db.Model):
    __tablename__ = 'courses'
    id = db.Column(db.Integer, primary_key=True)
    course_code = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    classrooms = db.relationship('Classroom', backref='course', lazy='dynamic', cascade="all, delete-orphan")
    __table_args__ = (db.UniqueConstraint('course_code', 'department_id', name='_course_dept_uc'),)

# --- Dynamic Rubric Engine Models ---

class Rubric(db.Model):
    __tablename__ = 'rubrics'
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignments.id'), unique=True, nullable=False)
    items = db.relationship('RubricItem', backref='rubric', lazy='dynamic', cascade="all, delete-orphan")

class RubricItem(db.Model):
    __tablename__ = 'rubric_items'
    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey('rubrics.id'), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    points = db.Column(db.Float, nullable=False, default=0.0)
    applied_to_submissions = db.relationship('AppliedRubricItem', backref='rubric_item', lazy='dynamic', cascade="all, delete-orphan")

class AppliedRubricItem(db.Model):
    __tablename__ = 'applied_rubric_items'
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submissions.id'), nullable=False)
    rubric_item_id = db.Column(db.Integer, db.ForeignKey('rubric_items.id'), nullable=False)
    __table_args__ = (db.UniqueConstraint('submission_id', 'rubric_item_id', name='_submission_item_uc'),)

# Association table for many-to-many between Teacher and Department
teacher_department_association = db.Table('teacher_department_association',
    db.Column('teacher_id', db.Integer, db.ForeignKey('teachers.id'), primary_key=True),
    db.Column('department_id', db.Integer, db.ForeignKey('departments.id'), primary_key=True)
)

# --- Refactored User and Classroom Models ---

class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    full_name = db.Column(db.String(120), nullable=True)
    role = db.Column(db.String(50), default='admin')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    roll_number = db.Column(db.String(50), unique=True, nullable=False)
    profile_image = db.Column(db.String(200), nullable=True)
    is_approved = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    submissions = db.relationship('Submission', backref='student', lazy='dynamic', cascade="all, delete-orphan")
    enrollments = db.relationship('Enrollment', backref='student', lazy='dynamic', cascade="all, delete-orphan")
    documents = db.relationship('StudentDocument', backref='student', lazy='dynamic', cascade="all, delete-orphan")

class Teacher(db.Model):
    __tablename__ = 'teachers'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True)
    profile_image = db.Column(db.String(200), nullable=True)
    is_approved = db.Column(db.Boolean, default=False)
    approved_by = db.Column(db.Integer, db.ForeignKey('admins.id'), nullable=True)
    approved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    classrooms = db.relationship('Classroom', backref='teacher', lazy='dynamic', cascade="all, delete-orphan")
    assignments = db.relationship('Assignment', backref='teacher', lazy='dynamic', cascade="all, delete-orphan")
    departments = db.relationship('Department', secondary=teacher_department_association, backref=db.backref('teachers', lazy='dynamic'))
    documents = db.relationship('TeacherDocument', backref='teacher', lazy='dynamic', cascade="all, delete-orphan")

def generate_invite_code():
    return str(uuid.uuid4().hex)[:8].upper()

class Classroom(db.Model):
    __tablename__ = 'classrooms'
    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    section_name = db.Column(db.String(100), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    description = db.Column(db.Text, nullable=True)
    invite_code = db.Column(db.String(8), unique=True, nullable=False, default=generate_invite_code)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assignments = db.relationship('Assignment', backref='classroom', lazy='dynamic', cascade="all, delete-orphan")
    enrollments = db.relationship('Enrollment', backref='classroom', lazy='dynamic', cascade="all, delete-orphan")

class Assignment(db.Model):
    __tablename__ = 'assignments'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    submission_deadline = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    max_score = db.Column(db.Float, default=10.0)
    submissions = db.relationship('Submission', backref='assignment', lazy='dynamic', cascade="all, delete-orphan")
    rubric = db.relationship('Rubric', uselist=False, backref='assignment', cascade="all, delete-orphan")
    clusters = db.relationship('Cluster', backref='assignment', lazy='dynamic', cascade="all, delete-orphan")

class Cluster(db.Model):
    __tablename__ = 'clusters'
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignments.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False) # e.g., "Cluster 1"
    conceptual_summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    submissions = db.relationship('ClusteredSubmission', backref='cluster', lazy='dynamic', cascade="all, delete-orphan")

class ClusteredSubmission(db.Model):
    __tablename__ = 'clustered_submissions'
    id = db.Column(db.Integer, primary_key=True)
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=False)
    submission_id = db.Column(db.Integer, db.ForeignKey('submissions.id'), nullable=False, unique=True)
    submission = db.relationship('Submission', backref=db.backref('cluster_membership', uselist=False, cascade="all, delete-orphan"))

class Submission(db.Model):
    __tablename__ = 'submissions'
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignments.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    answer_text = db.Column(db.Text, nullable=True)
    submission_file_path = db.Column(db.String(255), nullable=True)
    submitted_at = db.Column(db.DateTime, nullable=True)
    evaluated_at = db.Column(db.DateTime, nullable=True)
    score = db.Column(db.Float, nullable=True)
    gemini_comment = db.Column(db.Text, nullable=True)
    group_id = db.Column(db.Integer, nullable=True, index=True)
    applied_rubrics = db.relationship('AppliedRubricItem', backref='submission', lazy='dynamic', cascade="all, delete-orphan")

    @property
    def total_score(self):
        return db.session.query(func.sum(RubricItem.points)).join(AppliedRubricItem).filter(AppliedRubricItem.submission_id == self.id).scalar() or 0.0

class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_approved = db.Column(db.Boolean, default=False)
    approved_by = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=True)
    approved_at = db.Column(db.DateTime, nullable=True)
    __table_args__ = (db.UniqueConstraint('student_id', 'classroom_id', name='_enrollment_stud_classroom_uc'),)

class StudentDocument(db.Model):
    __tablename__ = 'student_documents'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    document_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class TeacherDocument(db.Model):
    __tablename__ = 'teacher_documents'
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    document_type = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


def recalculate_scores_after_rubric_update(mapper, connection, target):
    points_history = get_history(target, 'points')
    if not points_history.has_changes():
        return

    submissions_to_update = db.session.query(AppliedRubricItem.submission_id)\
                                      .filter(AppliedRubricItem.rubric_item_id == target.id)\
                                      .all()
    submission_ids = [s.submission_id for s in submissions_to_update]

    if not submission_ids:
        return

    for sub_id in submission_ids:
        submission = Submission.query.get(sub_id)
        if submission:
            new_score = submission.total_score
            connection.execute(
                db.text("UPDATE submissions SET score = :new_score WHERE id = :sub_id"),
                [{'new_score': new_score, 'sub_id': sub_id}]
            )

event.listen(RubricItem, 'after_update', recalculate_scores_after_rubric_update)
