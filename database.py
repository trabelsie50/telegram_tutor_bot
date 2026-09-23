import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from config import get_settings


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "tutor_bot.db"
)


class StudentDB:
    """
    طبقة الوصول إلى قاعدة بيانات SQLite لتتبع بيانات التلاميذ،
    نتائج الاختبارات، أنواع الأخطاء الشائعة، وتقدم التعلم.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_data_directory()
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_tables()

    def _ensure_data_directory(self):
        data_dir = os.path.dirname(self.db_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

    def _create_tables(self):
        cursor = self._connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id   INTEGER PRIMARY KEY,
                name         TEXT    NOT NULL DEFAULT '',
                grade        TEXT    NOT NULL DEFAULT 'السنة الخامسة ابتدائي',
                class_name   TEXT    NOT NULL DEFAULT '',
                created_at   TEXT    NOT NULL,
                updated_at   TEXT    NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id     INTEGER NOT NULL,
                lesson_name    TEXT    NOT NULL,
                subject        TEXT    NOT NULL,
                score          INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                date_taken     TEXT    NOT NULL,
                details        TEXT    DEFAULT '{}',
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_types (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id     INTEGER NOT NULL,
                error_category TEXT    NOT NULL,
                lesson_name    TEXT    NOT NULL,
                subject        TEXT    NOT NULL,
                count          INTEGER NOT NULL DEFAULT 1,
                first_occurred TEXT    NOT NULL,
                last_occurred  TEXT    NOT NULL,
                UNIQUE (student_id, error_category, lesson_name, subject),
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                student_id    INTEGER NOT NULL,
                subject       TEXT    NOT NULL,
                lesson_name   TEXT    NOT NULL,
                completed     INTEGER DEFAULT 0,
                avg_score     REAL    DEFAULT 0.0,
                attempts      INTEGER DEFAULT 0,
                last_activity TEXT    NOT NULL,
                PRIMARY KEY (student_id, subject, lesson_name)
            )
        """)

        self._connection.commit()

    # ------------------------------------------------------------------ #
    #  إدارة الاتصال
    # ------------------------------------------------------------------ #

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------ #
    #  التلاميذ
    # ------------------------------------------------------------------ #

    def add_or_update_student(
        self,
        student_id: int,
        name: str,
        grade: str = "السنة الخامسة ابتدائي",
        class_name: str = "",
    ) -> bool:
        now = datetime.now().isoformat()
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT student_id FROM students WHERE student_id = ?",
            (student_id,),
        )
        if cursor.fetchone():
            cursor.execute(
                """
                UPDATE students
                   SET name = ?, updated_at = ?
                 WHERE student_id = ?
                """,
                (name, now, student_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO students
                    (student_id, name, grade, class_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (student_id, name, grade, class_name, now, now),
            )
        self._connection.commit()
        return True

    def get_student(self, student_id: int) -> Optional[Dict[str, Any]]:
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------ #
    #  نتائج الاختبارات
    # ------------------------------------------------------------------ #

    def record_quiz(
        self,
        student_id: int,
        lesson_name: str,
        subject: str,
        score: int,
        total_questions: int,
        details: Optional[Dict[str, Any]] = None,
        error_categories: Optional[List[str]] = None,
    ) -> int:
        now = datetime.now().isoformat()
        details_json = json.dumps(details or {})
        cursor = self._connection.cursor()

        cursor.execute(
            """
            INSERT INTO quiz_results
                (student_id, lesson_name, subject, score, total_questions,
                 date_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                lesson_name,
                subject,
                score,
                total_questions,
                now,
                details_json,
            ),
        )
        quiz_row_id = cursor.lastrowid

        if error_categories:
            for category in error_categories:
                self._insert_error_type(
                    student_id,
                    category,
                    lesson_name,
                    subject,
                    now,
                )

        self._connection.commit()
        return quiz_row_id

    def get_quiz_history(
        self, student_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT * FROM quiz_results
             WHERE student_id = ?
            ORDER BY date_taken DESC
            LIMIT ?
            """,
            (student_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------ #
    #  أنواع الأخطاء الشائعة
    # ------------------------------------------------------------------ #

    def _insert_error_type(
        self,
        student_id: int,
        error_category: str,
        lesson_name: str,
        subject: str,
        now: str,
    ) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO error_types
                (student_id, error_category, lesson_name, subject,
                 count, first_occurred, last_occurred)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(student_id, error_category, lesson_name, subject)
            DO UPDATE SET
                count = count + 1,
                last_occurred = ?
            """,
            (
                student_id,
                error_category,
                lesson_name,
                subject,
                now,
                now,
                now,
            ),
        )

    def record_error_type(
        self,
        student_id: int,
        error_category: str,
        lesson_name: str,
        subject: str,
    ) -> bool:
        now = datetime.now().isoformat()
        self._insert_error_type(
            student_id, error_category, lesson_name, subject, now
        )
        self._connection.commit()
        return True

    def get_common_errors(
        self, student_id: int, limit: int = 10
    ) -> List[Dict[str, Any]]:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT error_category, lesson_name, subject, count,
                   first_occurred, last_occurred
              FROM error_types
             WHERE student_id = ?
            ORDER BY count DESC
            LIMIT ?
            """,
            (student_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------ #
    #  تقدّم التعلّم
    # ------------------------------------------------------------------ #

    def update_learning_progress(
        self,
        student_id: int,
        subject: str,
        lesson_name: str,
        score: Optional[float] = None,
        completed: bool = True,
    ) -> bool:
        now = datetime.now().isoformat()
        cursor = self._connection.cursor()

        cursor.execute(
            """
            SELECT * FROM learning_progress
             WHERE student_id = ? AND subject = ? AND lesson_name = ?
            """,
            (student_id, subject, lesson_name),
        )
        row = cursor.fetchone()

        if row is not None:
            existing = dict(row)
            new_attempts = existing["attempts"] + 1
            if score is not None:
                new_avg = (
                    (existing["avg_score"] * existing["attempts"] + score)
                    / new_attempts
                )
            else:
                new_avg = existing["avg_score"]
            cursor.execute(
                """
                UPDATE learning_progress
                   SET completed = ?, avg_score = ?, attempts = ?,
                       last_activity = ?
                 WHERE student_id = ? AND subject = ? AND lesson_name = ?
                """,
                (
                    1 if completed else 0,
                    round(new_avg, 2),
                    new_attempts,
                    now,
                    student_id,
                    subject,
                    lesson_name,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO learning_progress
                    (student_id, subject, lesson_name, completed,
                     avg_score, attempts, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    subject,
                    lesson_name,
                    1 if completed else 0,
                    score if score is not None else 0.0,
                    1,
                    now,
                ),
            )

        self._connection.commit()
        return True

    def get_student_progress(self, student_id: int) -> Dict[str, Any]:
        cursor = self._connection.cursor()

        cursor.execute(
            """
            SELECT subject, lesson_name, completed, avg_score,
                   attempts, last_activity
              FROM learning_progress
             WHERE student_id = ?
            ORDER BY subject, lesson_name
            """,
            (student_id,),
        )
        lessons = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT subject,
                   COUNT(*) AS lesson_count,
                   SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END)
                       AS completed_count,
                   ROUND(AVG(avg_score), 2) AS overall_avg
              FROM learning_progress
             WHERE student_id = ?
             GROUP BY subject
            """,
            (student_id,),
        )
        subjects = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT COUNT(*) AS total FROM quiz_results WHERE student_id = ?",
            (student_id,),
        )
        total_quizzes = cursor.fetchone()["total"]

        cursor.execute(
            """
            SELECT subject, COUNT(*) AS quiz_count
              FROM quiz_results
             WHERE student_id = ?
             GROUP BY subject
            """,
            (student_id,),
        )
        quizzes_by_subject = {
            row["subject"]: row["quiz_count"] for row in cursor.fetchall()
        }

        return {
            "student_id": student_id,
            "total_quizzes": total_quizzes,
            "quizzes_by_subject": quizzes_by_subject,
            "subjects_summary": subjects,
            "lessons_detail": lessons,
        }

    # ------------------------------------------------------------------ #
    #  تقارير الأولياء
    # ------------------------------------------------------------------ #

    def generate_parent_report(self, student_id: int) -> Dict[str, Any]:
        student = self.get_student(student_id)
        if student is None:
            return {"error": "التلميذ غير موجود في قاعدة البيانات"}

        progress = self.get_student_progress(student_id)
        common_errors = self.get_common_errors(student_id, limit=5)
        quiz_history = self.get_quiz_history(student_id, limit=10)

        total_quizzes = len(quiz_history)
        if total_quizzes > 0:
            total_score = sum(q["score"] for q in quiz_history)
            max_possible = sum(q["total_questions"] for q in quiz_history)
            if max_possible > 0:
                overall_percentage = round(
                    total_score / max_possible * 100, 1
                )
            else:
                overall_percentage = 0.0
            avg_quiz_score = round(total_score / total_quizzes, 1)
        else:
            overall_percentage = 0.0
            avg_quiz_score = 0.0

        recent_by_subject: Dict[str, List[Dict[str, Any]]] = {}
        for q in quiz_history[:5]:
            subj = q["subject"]
            if subj not in recent_by_subject:
                recent_by_subject[subj] = []
            recent_by_subject[subj].append(
                {
                    "lesson": q["lesson_name"],
                    "score": q["score"],
                    "total": q["total_questions"],
                    "date": q["date_taken"],
                }
            )

        completed_lessons = [
            l for l in progress["lessons_detail"] if l["completed"] == 1
        ]
        total_lessons_tracked = len(progress["lessons_detail"])

        return {
            "student_name": student["name"],
            "student_id": student_id,
            "grade": student["grade"],
            "class": student["class_name"],
            "total_quizzes_taken": total_quizzes,
            "overall_percentage": overall_percentage,
            "average_quiz_score": avg_quiz_score,
            "recent_quizzes": recent_by_subject,
            "completed_lessons": completed_lessons,
            "total_lessons_tracked": total_lessons_tracked,
        }

    def get_student_by_id(self, student_id: int) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM students WHERE student_id = ?"
        row = self._connection.execute(query, (student_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_students_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        query = "SELECT * FROM students WHERE class_name = ? ORDER BY name"
        rows = self._connection.execute(query, (class_name,)).fetchall()
        return [dict(row) for row in rows]

    def update_student(self, student_id: int, data: Dict[str, Any]) -> bool:
        allowed_fields = [
            "name",
            "grade",
            "class_name",
        ]
        fields = [f for f in allowed_fields if f in data]
        if not fields:
            return False
        set_clause = ", ".join(f"{f} = ?" for f in fields)
        values = [data[f] for f in fields]
        values.append(student_id)
        query = f"UPDATE students SET {set_clause} WHERE student_id = ?"
        self._connection.execute(query, values)
        self._connection.commit()
        return True

    def delete_student(self, student_id: int) -> bool:
        cursor = self._connection.cursor()
        cursor.execute(
            "DELETE FROM students WHERE student_id = ?", (student_id,)
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def get_analytics_summary(self) -> Dict[str, Any]:
        cursor = self._connection.cursor()

        cursor.execute("SELECT COUNT(*) AS c FROM students")
        total_students = cursor.fetchone()["c"]

        cursor.execute(
            "SELECT COUNT(*) AS c FROM students WHERE grade = 'السنة الخامسة ابتدائي'"
        )
        active_students = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(*) AS c FROM quiz_results")
        total_quizzes = cursor.fetchone()["c"]

        cursor.execute("SELECT COUNT(DISTINCT subject) AS c FROM quiz_results")
        total_subjects = cursor.fetchone()["c"]

        cursor.execute(
            "SELECT COUNT(DISTINCT lesson_name) AS c FROM quiz_results"
        )
        total_lessons = cursor.fetchone()["c"]

        cursor.execute("SELECT AVG(score) AS avg_score FROM quiz_results")
        avg_result = cursor.fetchone()
        avg_score = round(avg_result["avg_score"], 1) if avg_result["avg_score"] else 0.0

        cursor.execute(
            """
            SELECT lesson_name, AVG(score) AS avg_s
            FROM quiz_results
            GROUP BY lesson_name
            ORDER BY avg_s DESC
            LIMIT 5
            """
        )
        top_subjects = []
        for row in cursor.fetchall():
            top_subjects.append(
                {"subject": row["lesson_name"], "average_score": round(row["avg_s"], 1)}
            )

        return {
            "total_students": total_students,
            "active_students": active_students,
            "total_quizzes": total_quizzes,
            "total_subjects": total_subjects,
            "total_lessons": total_lessons,
            "average_quiz_score": avg_score,
            "top_subjects": top_subjects,
        }


# Alias to resolve forward reference issue
SchoolDatabase = StudentDB


def get_db_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data", "school.db")


def init_db() -> None:
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    SchoolDatabase(db_path)


def get_database() -> SchoolDatabase:
    db_path = get_db_path()
    return SchoolDatabase(db_path)


database = SchoolDatabase(get_db_path())
