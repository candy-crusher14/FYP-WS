import click
from flask.cli import with_appcontext
from extensions import db
from models import (University, Department, Course, Teacher, Student, Classroom, Assignment, 
                    Rubric, RubricItem, Submission, Enrollment, AppliedRubricItem, Admin, Cluster, ClusteredSubmission)
import os
import hashlib
from datetime import datetime, timedelta
from sqlalchemy import text

def hash_password(password): 
    return hashlib.sha256(password.encode()).hexdigest()

@click.command('seed-test-data')
@with_appcontext
def seed_test_data():
    """Wipes and seeds the database with a full testing environment."""
    print("Wiping all existing test data...")
    # In a production environment with foreign keys, this order is crucial.
    # For SQLite, `db.drop_all()` and `db.create_all()` is more robust.
    try:
        db.drop_all()
        db.create_all()
        print("All tables dropped and recreated.")
    except Exception as e:
        print(f"Error wiping database: {e}")
        db.session.rollback()
    db.session.query(Submission).delete()
    db.session.query(AppliedRubricItem).delete()
    db.session.query(RubricItem).delete()
    db.session.query(Rubric).delete()
    db.session.query(Assignment).delete()
    db.session.query(Classroom).delete()
    db.session.query(Course).delete()
    # Clear the association table
    db.session.execute(text('DELETE FROM teacher_department_association'))
    db.session.query(Department).delete()
    db.session.query(University).delete()
    db.session.query(Student).delete()
    db.session.query(Teacher).delete()
    db.session.query(Admin).delete()
    db.session.commit()
    print("All tables wiped.")

    print("Starting database seeding for test environment...")

    # 1. Identify Existing Data: Pick one University
    uni_name = 'University of Mirpur Khas'
    university = University.query.filter_by(name=uni_name).first()
    if not university:
        university = University(name=uni_name)
        db.session.add(university)
        db.session.commit()
        print(f"Created University: {university.name}")

    # 2. Build the Test Branch: Department
    dept_name = 'Computer Science'
    department = Department.query.filter_by(name=dept_name, university_id=university.id).first()
    if not department:
        department = Department(name=dept_name, university_id=university.id)
        db.session.add(department)
        db.session.commit()
        print(f"Created Department: {department.name}")

    # 3. Create Teacher
    teacher_username = 'test_teacher'
    teacher = Teacher.query.filter_by(username=teacher_username).first()
    if not teacher:
        teacher = Teacher(
            username=teacher_username, 
            password=hash_password(os.getenv('DEFAULT_TEACHER_PASS', 'teacher123')),
            full_name='Dr. Test Instructor',
            email='teacher@test.edu',
            is_approved=True
        )
        teacher.departments.append(department)
        db.session.add(teacher)
        db.session.commit()
        print(f"Created Teacher: {teacher.username}")

    # 4. Create Course
    course_code = 'CS101'
    course = Course.query.filter_by(course_code=course_code, department_id=department.id).first()
    if not course:
        course = Course(course_code=course_code, title='Introduction to Programming', department_id=department.id)
        db.session.add(course)
        db.session.commit()
        print(f"Created Course: {course.title}")

    # 5. Create Classroom
    classroom = Classroom.query.filter_by(course_id=course.id, teacher_id=teacher.id).first()
    if not classroom:
        classroom = Classroom(course_id=course.id, teacher_id=teacher.id, section_name='Fall 2025 - Section A')
        db.session.add(classroom)
        db.session.commit()
        print(f"Created Classroom: {classroom.section_name}")

    # 6. Create Assignment for Semantic Test
    assignment_title = 'Physics Question: Gravity'
    assignment = Assignment.query.filter_by(title=assignment_title, classroom_id=classroom.id).first()
    if not assignment:
        assignment = Assignment(
            title=assignment_title,
            classroom_id=classroom.id,
            teacher_id=teacher.id,
            instructions='What is gravity?',
            submission_deadline=datetime.utcnow() + timedelta(days=7),
            max_score=10.0
        )
        db.session.add(assignment)
        db.session.commit()
        print(f"Created Assignment: {assignment.title}")

    # 7. Create 5 unique students and their submissions for the semantic test
    students_and_answers = [
        {'name': 'Student A', 'username': 'student_a', 'answer': 'Gravity pulls things down.'},
        {'name': 'Student B', 'username': 'student_b', 'answer': 'It is the force of attraction from the earth.'},
        {'name': 'Student C', 'username': 'student_c', 'answer': 'It involves magnetic poles and north.'},
        {'name': 'Student D', 'username': 'student_d', 'answer': 'I don\'t know.'},
        {'name': 'Student E', 'username': 'student_e', 'answer': 'Gravity is why things fall.'}
    ]

    # Clear existing submissions for this assignment to ensure a clean test
    Submission.query.filter_by(assignment_id=assignment.id).delete()
    db.session.commit()

    for student_data in students_and_answers:
        # Create student if they don't exist
        student = Student.query.filter_by(username=student_data['username']).first()
        if not student:
            student = Student(
                username=student_data['username'],
                password=hash_password('student123'),
                full_name=student_data['name'],
                email=f"{student_data['username']}@test.edu",
                roll_number=f"TEST_{student_data['username'].upper()}"
            )
            db.session.add(student)
            db.session.commit()
            print(f"Created Student: {student.full_name}")

        # Enroll student if not already enrolled
        enrollment = Enrollment.query.filter_by(student_id=student.id, classroom_id=classroom.id).first()
        if not enrollment:
            enrollment = Enrollment(student_id=student.id, classroom_id=classroom.id, is_approved=True)
            db.session.add(enrollment)
            db.session.commit()
            print(f"Enrolled {student.full_name} in {classroom.section_name}")
        
        # Create submission
        submission = Submission(
            assignment_id=assignment.id,
            student_id=student.id,
            answer_text=student_data['answer'],
            submitted_at=datetime.utcnow()
        )
        db.session.add(submission)

    db.session.commit()
    print(f"Created {len(students_and_answers)} unique submissions for the semantic test.")
    print("\n--------------------------------------------------")
    print(f"[SUCCESS] Database seeding complete.")
    print(f"--> Test Assignment ID created: {assignment.id}")
    print("--> Use this ID for the /dev/test-cluster/<id> route.")
    print("--------------------------------------------------")

@click.command('seed-demo')
@with_appcontext
def seed_demo():
    """Wipes and seeds the database with the 'Golden Demo' state."""
    print("--- Starting Golden Demo Seeding ---")
    try:
        db.drop_all()
        db.create_all()
        print("Database wiped and recreated.")

        # 1. Create University and Department
        uni = University(name='Demo University')
        db.session.add(uni)
        db.session.commit()
        dept = Department(name='Data Science', university_id=uni.id)
        db.session.add(dept)
        db.session.commit()
        print("Created Demo University and Data Science Department.")

        # 2. Create Demo Accounts
        admin = Admin(username='admin_demo', email='admin@demo.com', password=hash_password('password123'), full_name='Demo Admin')
        teacher = Teacher(username='teacher_demo', email='teacher@demo.com', password=hash_password('password123'), full_name='Dr. Demo', university_id=uni.id, is_approved=True)
        teacher.departments.append(dept)
        student_user = Student(username='student_demo', email='student@demo.com', password=hash_password('password123'), full_name='Demo Student', university=uni.name, department=dept.name, year=3, roll_number='DEMO001')
        db.session.add_all([admin, teacher, student_user])
        db.session.commit()
        print("Created Golden Accounts: admin@demo.com, teacher@demo.com, student@demo.com")

        # 3. Create Classroom and Assignment
        classroom = Classroom(name='Data Science 101', teacher_id=teacher.id, department=dept.name, year=3, semester='Semester 5')
        db.session.add(classroom)
        db.session.commit()

        assignment = Assignment(
            title='Neural Networks & Logic',
            classroom_id=classroom.id,
            teacher_id=teacher.id,
            instructions='Explain the difference between a perceptron and a multi-layer perceptron (MLP). What role does an activation function play?',
            submission_deadline=datetime.utcnow() + timedelta(days=10),
            max_score=20.0
        )
        db.session.add(assignment)
        db.session.commit()
        print(f"Created Classroom '{classroom.name}' and Assignment '{assignment.title}'.")

        # 4. Create 10 varied student submissions
        student_answers = [
            {'name': 'Alex', 'answer': 'A perceptron is a single neuron, but an MLP has many layers of them. The activation function decides if the neuron fires.'},
            {'name': 'Brenda', 'answer': 'A perceptron is just one layer. An MLP is a deep neural network with hidden layers. Activation functions like ReLU add non-linearity so it can learn complex things.'},
            {'name': 'Carl', 'answer': 'MLP is multiple perceptrons. The activation function is the part that does the math to get the output.'},
            {'name': 'Dana', 'answer': 'A perceptron is a linear classifier. A multi-layer perceptron can learn non-linear boundaries because it has hidden layers and non-linear activation functions.'},
            {'name': 'Eric', 'answer': 'The perceptron is the basic building block. You stack them to make an MLP. The activation function squashes the output to be between 0 and 1.'},
            {'name': 'Fiona', 'answer': 'I am not sure about the perceptron, but an MLP is used for deep learning. The activation function activates the neuron.'},
            {'name': 'George', 'answer': 'A perceptron can only solve linearly separable problems. An MLP with non-linear activations can solve more complex problems. The activation introduces non-linearity.'},
            {'name': 'Hannah', 'answer': 'A perceptron is one layer, an MLP is many. The activation function is like a switch.'},
            {'name': 'Ian', 'answer': 'Multi-layer perceptrons have an input layer, hidden layers, and an output layer. Perceptrons do not have hidden layers. The activation function determines the output of the neuron.'},
            {'name': 'Jenna', 'answer': 'An MLP is a feedforward neural network with multiple layers. A perceptron is the simplest form of this. The activation function is what allows it to learn.'}
        ]
        
        submissions = []
        # Add the main demo student to the list of submissions to be created
        all_students_for_submissions = [(student_user, {'answer': 'A perceptron is a single-layer neural network, whereas a multi-layer perceptron (MLP) has multiple layers, including hidden layers. The activation function introduces non-linearity, allowing the MLP to learn complex patterns that a simple perceptron cannot.'})]

        for i, data in enumerate(student_answers):
            student = Student(username=f'demo_student_{i}', email=f'demo{i}@demo.edu', password=hash_password('password123'), full_name=data['name'], university=uni.name, department=dept.name, year=3, roll_number=f'DEMO{100+i}')
            all_students_for_submissions.append((student, data))

        for student, data in all_students_for_submissions:
            db.session.add(student)
            db.session.commit()
            enrollment = Enrollment(student_id=student.id, classroom_id=classroom.id, is_approved=True)
            db.session.add(enrollment)
            submission = Submission(assignment_id=assignment.id, student_id=student.id, answer_text=data['answer'], submitted_at=datetime.utcnow() - timedelta(hours=i))
            submissions.append(submission)
            db.session.add(submission)
        
        db.session.commit()
        print(f"Seeded {len(student_answers)} students and submissions.")

        # 5. Pre-generate and save clusters
        cluster1 = Cluster(assignment_id=assignment.id, name='Cluster 1 (Correct)', conceptual_summary='Correctly identifies that an MLP has hidden layers and that activation functions add non-linearity.')
        cluster2 = Cluster(assignment_id=assignment.id, name='Cluster 2 (Partially Correct)', conceptual_summary='Understands the basic layered structure but is vague on the role of activation functions.')
        cluster3 = Cluster(assignment_id=assignment.id, name='Cluster 3 (Incorrect/Confused)', conceptual_summary='Confuses MLPs with other concepts or has fundamental misunderstandings.')
        db.session.add_all([cluster1, cluster2, cluster3])
        db.session.commit()

        # Map submissions to clusters
        cluster_mappings = {
            cluster1: [1, 3, 6, 9],  # Brenda, Dana, George, Jenna
            cluster2: [0, 2, 8],      # Alex, Carl, Ian
            cluster3: [4, 5, 7]       # Eric, Fiona, Hannah
        }

        # The main 'student_demo' will be the first in the submissions list
        main_demo_submission_index = 0 

        for cluster, submission_indices in cluster_mappings.items():
            for index in submission_indices:
                # Adjust index to account for the main demo student added at the beginning
                submission_to_cluster = submissions[index + 1] 
                clustered_sub = ClusteredSubmission(cluster_id=cluster.id, submission_id=submission_to_cluster.id)
                db.session.add(clustered_sub)
        
        # Add the main demo student's submission to the correct cluster
        main_demo_submission = submissions[main_demo_submission_index]
        db.session.add(ClusteredSubmission(cluster_id=cluster1.id, submission_id=main_demo_submission.id))

        db.session.commit()
        print("Manually created and linked 3 clusters for the demo assignment.")

        print("--- Golden Demo Seeding Complete! ---")

    except Exception as e:
        db.session.rollback()
        print(f"An error occurred during demo seeding: {e}")
