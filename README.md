# AI-Powered Assignment Evaluator

This web application is a sophisticated tool for educators, leveraging AI to streamline the grading process for open-ended, text-based assignments. It uses semantic analysis to group student submissions by conceptual understanding, allowing teachers to grade entire groups of similar answers at once.

## Key Features

- **AI-Assisted Grading:** Automatically groups student submissions into conceptual clusters using Google's Gemini API.
- **Semantic Clustering:** Goes beyond simple keyword matching to understand the meaning and intent behind student answers.
- **Bulk Grading:** Grade entire clusters of students at once, with the ability to provide group-specific feedback.
- **Manual Override:** Teachers retain full control, with the ability to move students between clusters or override individual grades.
- **Dynamic Classroom Management:** Teachers can create classrooms, manage student enrollments, and generate unique invite codes.
- **Comprehensive Dashboards:** Separate, feature-rich dashboards for students, teachers, and admins.
- **Data Export:** Export assignment grades to CSV for easy record-keeping.

## Getting Started

### Prerequisites

- Python 3.10+
- An environment variable manager (e.g., `python-dotenv`)
- A Google Gemini API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    - Create a `.env` file in the project root.
    - Add the following variables:
      ```
      SECRET_KEY='a_strong_and_random_secret_key'
      GEMINI_API_KEY='your_google_gemini_api_key'
      ADMIN_SECRET='a_secure_admin_code_for_registration'
      ```

### Running the Application

1.  **Initialize the database:**
    ```bash
    flask db init
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```

2.  **Seed the database for the Golden Demo:**
    - This command will wipe the database and create a pre-configured set of users, classrooms, assignments, and AI clusters for a perfect demonstration.
    ```bash
    flask seed-demo
    ```

3.  **Run the Flask application:**
    ```bash
    flask run
    ```

## Golden Demo Credentials

After running the `seed-demo` command, you can use the following credentials to explore the application:

- **Teacher:**
  - **Username:** `teacher_demo`
  - **Password:** `password123`
- **Student:**
  - **Username:** `student_demo`
  - **Password:** `password123`
- **Admin:**
  - **Username:** `admin_demo`
  - **Password:** `password123`
  - **Admin Code:** The value you set for `ADMIN_SECRET` in your `.env` file (defaults to `admin123` if not set).
