"""Database management for one-pixel attack experiments."""

import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, connection_string: str = None):
        """Initialize database manager with connection string."""
        self.connection_string = connection_string or os.environ.get('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("DATABASE_URL not found in environment variables")
        self.pool = SimpleConnectionPool(1, 5, self.connection_string)
        self._create_tables()
    
    def _get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Questions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rex_vqa_onepix_questions (
                        id SERIAL PRIMARY KEY,
                        study_id VARCHAR(255) UNIQUE NOT NULL,
                        patient_id VARCHAR(255),
                        question TEXT NOT NULL,
                        options JSONB NOT NULL,
                        correct_answer CHAR(1) NOT NULL,
                        image_paths JSONB NOT NULL,
                        question_type VARCHAR(100),
                        pathology VARCHAR(100),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Results table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rex_vqa_onepix_results (
                        id SERIAL PRIMARY KEY,
                        run_id VARCHAR(255) NOT NULL,
                        study_id VARCHAR(255) NOT NULL,
                        attack_type VARCHAR(50) NOT NULL,
                        model_answer CHAR(1),
                        model_explanation TEXT,
                        model_confidence FLOAT,
                        model_findings JSONB,
                        is_correct BOOLEAN,
                        response_time FLOAT,
                        error_message TEXT,
                        parse_success BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (study_id) REFERENCES rex_vqa_onepix_questions(study_id),
                        UNIQUE(run_id, study_id, attack_type)
                    )
                """)
                
                # Attack log table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rex_vqa_onepix_log (
                        id SERIAL PRIMARY KEY,
                        run_id VARCHAR(255) NOT NULL,
                        study_id VARCHAR(255) NOT NULL,
                        image_path TEXT NOT NULL,
                        attack_type VARCHAR(50) NOT NULL,
                        original_prob FLOAT NOT NULL,
                        adversarial_prob FLOAT NOT NULL,
                        pixel_x_224 INTEGER NOT NULL,
                        pixel_y_224 INTEGER NOT NULL,
                        pixel_x_orig INTEGER NOT NULL,
                        pixel_y_orig INTEGER NOT NULL,
                        original_pixel_value INTEGER NOT NULL,
                        new_pixel_value INTEGER NOT NULL,
                        pixel_delta FLOAT NOT NULL,
                        in_lung_mask BOOLEAN NOT NULL,
                        lung_pixels_count INTEGER,
                        total_pixels INTEGER,
                        success BOOLEAN NOT NULL,
                        prob_change FLOAT NOT NULL,
                        iterations INTEGER NOT NULL,
                        fitness FLOAT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (study_id) REFERENCES rex_vqa_onepix_questions(study_id)
                    )
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_results_run_id ON rex_vqa_onepix_results(run_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_results_study_id ON rex_vqa_onepix_results(study_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_results_attack_type ON rex_vqa_onepix_results(attack_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_log_run_id ON rex_vqa_onepix_log(run_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_log_study_id ON rex_vqa_onepix_log(study_id)")
                
                conn.commit()
        finally:
            self._return_connection(conn)
    
    def insert_questions(self, questions_df: pd.DataFrame) -> int:
        """Insert questions into database, handling duplicates."""
        conn = self._get_connection()
        inserted = 0
        try:
            with conn.cursor() as cur:
                for _, row in questions_df.iterrows():
                    # Parse image paths if string
                    image_paths = row['image_path']
                    if isinstance(image_paths, str):
                        import ast
                        try:
                            image_paths = ast.literal_eval(image_paths)
                        except:
                            image_paths = [image_paths]
                    
                    # Parse options if needed
                    options = row['options']
                    if isinstance(options, str):
                        # Assume it's a formatted string like "A. ...\nB. ..."
                        options_list = [opt.strip() for opt in options.split('\n') if opt.strip()]
                        options = {opt[0]: opt[3:].strip() for opt in options_list if len(opt) > 3}
                    
                    cur.execute("""
                        INSERT INTO rex_vqa_onepix_questions 
                        (study_id, patient_id, question, options, correct_answer, 
                         image_paths, question_type, pathology, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (study_id) DO NOTHING
                    """, (
                        row.get('study_id'),
                        row.get('PatientID', row.get('patient_id')),
                        row['question'],
                        json.dumps(options),
                        row['correct_answer'],
                        json.dumps(image_paths),
                        row.get('question_type', 'medical'),
                        row.get('pathology', 'Pneumonia'),
                        json.dumps(row.get('metadata', {}))
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
                
                conn.commit()
        finally:
            self._return_connection(conn)
        
        return inserted
    
    def insert_result(self, run_id: str, result: Dict[str, Any]) -> bool:
        """Insert a single evaluation result."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rex_vqa_onepix_results 
                    (run_id, study_id, attack_type, model_answer, model_explanation,
                     model_confidence, model_findings, is_correct, response_time,
                     error_message, parse_success)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, study_id, attack_type) 
                    DO UPDATE SET
                        model_answer = EXCLUDED.model_answer,
                        model_explanation = EXCLUDED.model_explanation,
                        model_confidence = EXCLUDED.model_confidence,
                        model_findings = EXCLUDED.model_findings,
                        is_correct = EXCLUDED.is_correct,
                        response_time = EXCLUDED.response_time,
                        error_message = EXCLUDED.error_message,
                        parse_success = EXCLUDED.parse_success,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    run_id,
                    result['study_id'],
                    result['attack_type'],
                    result.get('model_answer'),
                    result.get('model_explanation'),
                    result.get('model_confidence'),
                    json.dumps(result.get('model_findings', [])),
                    result.get('model_answer') == result.get('correct_answer'),
                    result.get('response_time'),
                    result.get('error_message'),
                    result.get('parse_success', True)
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error inserting result: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)
    
    def insert_attack_log(self, run_id: str, log_entry: Dict[str, Any]) -> bool:
        """Insert attack log entry."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Convert numpy types to Python native types
                in_lung_mask = bool(log_entry['in_lung_mask']) if hasattr(log_entry['in_lung_mask'], 'item') else log_entry['in_lung_mask']
                
                cur.execute("""
                    INSERT INTO rex_vqa_onepix_log 
                    (run_id, study_id, image_path, attack_type, original_prob, adversarial_prob,
                     pixel_x_224, pixel_y_224, pixel_x_orig, pixel_y_orig,
                     original_pixel_value, new_pixel_value, pixel_delta, in_lung_mask,
                     lung_pixels_count, total_pixels, success, prob_change, iterations, fitness)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    run_id,
                    log_entry['study_id'],
                    log_entry['image_path'],
                    log_entry['attack_type'],
                    float(log_entry['original_prob']),
                    float(log_entry['adversarial_prob']),
                    int(log_entry['pixel_x_224']),
                    int(log_entry['pixel_y_224']),
                    int(log_entry['pixel_x_orig']),
                    int(log_entry['pixel_y_orig']),
                    int(log_entry['original_pixel_value']),
                    int(log_entry['new_pixel_value']),
                    float(log_entry['pixel_delta']),
                    in_lung_mask,
                    int(log_entry['lung_pixels_count']) if log_entry['lung_pixels_count'] is not None else None,
                    int(log_entry['total_pixels']),
                    bool(log_entry['success']) if hasattr(log_entry['success'], 'item') else log_entry['success'],
                    float(log_entry['prob_change']),
                    int(log_entry['iterations']),
                    float(log_entry['fitness'])
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error inserting attack log: {e}")
            conn.rollback()
            return False
        finally:
            self._return_connection(conn)
    
    def get_results_by_run(self, run_id: str, attack_type: Optional[str] = None) -> pd.DataFrame:
        """Get results for a specific run."""
        conn = self._get_connection()
        try:
            query = """
                SELECT r.*, q.question, q.options, q.correct_answer, q.image_paths
                FROM rex_vqa_onepix_results r
                JOIN rex_vqa_onepix_questions q ON r.study_id = q.study_id
                WHERE r.run_id = %s
            """
            params = [run_id]
            
            if attack_type:
                query += " AND r.attack_type = %s"
                params.append(attack_type)
            
            return pd.read_sql(query, conn, params=params)
        finally:
            self._return_connection(conn)
    
    def get_attack_logs_by_run(self, run_id: str) -> pd.DataFrame:
        """Get attack logs for a specific run."""
        conn = self._get_connection()
        try:
            query = """
                SELECT l.*, q.patient_id, q.pathology
                FROM rex_vqa_onepix_log l
                JOIN rex_vqa_onepix_questions q ON l.study_id = q.study_id
                WHERE l.run_id = %s
                ORDER BY l.created_at
            """
            return pd.read_sql(query, conn, params=[run_id])
        finally:
            self._return_connection(conn)
    
    def get_robustness_summary(self, run_id: str) -> Dict[str, Any]:
        """Calculate robustness metrics for a run."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get baseline and attack results
                cur.execute("""
                    WITH baseline AS (
                        SELECT study_id, model_answer, is_correct, model_confidence
                        FROM rex_vqa_onepix_results
                        WHERE run_id = %s AND attack_type = 'baseline'
                    ),
                    attack AS (
                        SELECT study_id, model_answer, is_correct, model_confidence
                        FROM rex_vqa_onepix_results
                        WHERE run_id = %s AND attack_type = 'onepixel'
                    )
                    SELECT 
                        COUNT(DISTINCT b.study_id) as total_samples,
                        CASE WHEN COUNT(b.study_id) > 0 
                            THEN COUNT(CASE WHEN b.is_correct THEN 1 END)::float / COUNT(b.study_id) 
                            ELSE 0 END as baseline_accuracy,
                        CASE WHEN COUNT(a.study_id) > 0 
                            THEN COUNT(CASE WHEN a.is_correct THEN 1 END)::float / COUNT(a.study_id) 
                            ELSE 0 END as attack_accuracy,
                        CASE WHEN COUNT(b.study_id) > 0 
                            THEN COUNT(CASE WHEN b.model_answer != a.model_answer THEN 1 END)::float / COUNT(b.study_id) 
                            ELSE 0 END as attack_success_rate,
                        CASE WHEN COUNT(CASE WHEN b.is_correct THEN 1 END) > 0 
                            THEN COUNT(CASE WHEN b.is_correct AND NOT a.is_correct THEN 1 END)::float / COUNT(CASE WHEN b.is_correct THEN 1 END) 
                            ELSE 0 END as targeted_success_rate,
                        AVG(b.model_confidence) as avg_confidence_baseline,
                        AVG(a.model_confidence) as avg_confidence_attack
                    FROM baseline b
                    LEFT JOIN attack a ON b.study_id = a.study_id
                """, (run_id, run_id))
                
                summary = dict(cur.fetchone())
                
                # Get attack statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_attacks,
                        COUNT(CASE WHEN success THEN 1 END) as successful_attacks,
                        AVG(prob_change) as avg_prob_change,
                        AVG(pixel_delta) as avg_pixel_delta,
                        CASE WHEN COUNT(*) > 0 
                            THEN COUNT(CASE WHEN in_lung_mask THEN 1 END)::float / COUNT(*) 
                            ELSE 0 END as lung_mask_ratio,
                        AVG(iterations) as avg_iterations
                    FROM rex_vqa_onepix_log
                    WHERE run_id = %s
                """, (run_id,))
                
                attack_stats = dict(cur.fetchone())
                summary.update(attack_stats)
                
                return summary
        finally:
            self._return_connection(conn)
    
    def close(self):
        """Close all database connections."""
        self.pool.closeall()