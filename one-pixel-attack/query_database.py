"""Query utilities for the one-pixel attack database."""

import argparse
import pandas as pd
from database import DatabaseManager


def list_runs(db_manager: DatabaseManager):
    """List all evaluation runs in the database."""
    conn = db_manager._get_connection()
    try:
        query = """
            SELECT 
                run_id,
                COUNT(DISTINCT study_id) as num_samples,
                COUNT(DISTINCT CASE WHEN attack_type = 'baseline' THEN study_id END) as baseline_count,
                COUNT(DISTINCT CASE WHEN attack_type = 'onepixel' THEN study_id END) as attack_count,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time
            FROM rex_vqa_onepix_results
            GROUP BY run_id
            ORDER BY MIN(created_at) DESC
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        db_manager._return_connection(conn)


def get_run_details(db_manager: DatabaseManager, run_id: str):
    """Get detailed results for a specific run."""
    print(f"\n=== Run Details: {run_id} ===")
    
    # Get summary
    summary = db_manager.get_robustness_summary(run_id)
    print("\nRobustness Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get per-question analysis
    conn = db_manager._get_connection()
    try:
        query = """
            WITH question_results AS (
                SELECT 
                    q.question,
                    COUNT(DISTINCT CASE WHEN r.attack_type = 'baseline' AND r.is_correct THEN r.study_id END) as baseline_correct,
                    COUNT(DISTINCT CASE WHEN r.attack_type = 'baseline' THEN r.study_id END) as baseline_total,
                    COUNT(DISTINCT CASE WHEN r.attack_type = 'onepixel' AND r.is_correct THEN r.study_id END) as attack_correct,
                    COUNT(DISTINCT CASE WHEN r.attack_type = 'onepixel' THEN r.study_id END) as attack_total
                FROM rex_vqa_onepix_results r
                JOIN rex_vqa_onepix_questions q ON r.study_id = q.study_id
                WHERE r.run_id = %s
                GROUP BY q.question
            )
            SELECT 
                question,
                baseline_correct::float / NULLIF(baseline_total, 0) as baseline_accuracy,
                attack_correct::float / NULLIF(attack_total, 0) as attack_accuracy,
                (baseline_correct::float / NULLIF(baseline_total, 0)) - 
                (attack_correct::float / NULLIF(attack_total, 0)) as accuracy_drop
            FROM question_results
            ORDER BY accuracy_drop DESC
            LIMIT 10
        """
        df = pd.read_sql(query, conn, params=[run_id])
        
        print("\nTop 10 Most Vulnerable Questions:")
        for idx, row in df.iterrows():
            print(f"\n{idx+1}. {row['question'][:80]}...")
            print(f"   Baseline Accuracy: {row['baseline_accuracy']:.2%}")
            print(f"   Attack Accuracy: {row['attack_accuracy']:.2%}")
            print(f"   Accuracy Drop: {row['accuracy_drop']:.2%}")
    finally:
        db_manager._return_connection(conn)


def export_results(db_manager: DatabaseManager, run_id: str, output_path: str):
    """Export all results for a run to CSV files."""
    import os
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Export results
    results_df = db_manager.get_results_by_run(run_id)
    results_df.to_csv(f"{output_path}/results_{run_id}.csv", index=False)
    print(f"Exported results to {output_path}/results_{run_id}.csv")
    
    # Export attack logs
    logs_df = db_manager.get_attack_logs_by_run(run_id)
    logs_df.to_csv(f"{output_path}/attack_logs_{run_id}.csv", index=False)
    print(f"Exported attack logs to {output_path}/attack_logs_{run_id}.csv")
    
    # Export comparison
    conn = db_manager._get_connection()
    try:
        query = """
            SELECT 
                b.study_id,
                q.question,
                q.correct_answer,
                b.model_answer as baseline_answer,
                a.model_answer as attack_answer,
                b.model_confidence as baseline_confidence,
                a.model_confidence as attack_confidence,
                b.is_correct as baseline_correct,
                a.is_correct as attack_correct,
                CASE WHEN b.model_answer != a.model_answer THEN TRUE ELSE FALSE END as answer_changed
            FROM rex_vqa_onepix_results b
            JOIN rex_vqa_onepix_results a ON b.study_id = a.study_id AND b.run_id = a.run_id
            JOIN rex_vqa_onepix_questions q ON b.study_id = q.study_id
            WHERE b.run_id = %s 
            AND b.attack_type = 'baseline' 
            AND a.attack_type = 'onepixel'
        """
        comparison_df = pd.read_sql(query, conn, params=[run_id])
        comparison_df.to_csv(f"{output_path}/comparison_{run_id}.csv", index=False)
        print(f"Exported comparison to {output_path}/comparison_{run_id}.csv")
    finally:
        db_manager._return_connection(conn)


def main():
    parser = argparse.ArgumentParser(description="Query the one-pixel attack database")
    parser.add_argument('command', choices=['list', 'details', 'export'],
                        help='Command to execute')
    parser.add_argument('--run-id', type=str,
                        help='Run ID for details or export commands')
    parser.add_argument('--output', type=str, default='./exports',
                        help='Output directory for export command')
    
    args = parser.parse_args()
    
    # Initialize database
    db_manager = DatabaseManager()
    
    try:
        if args.command == 'list':
            runs_df = list_runs(db_manager)
            print("\n=== Available Runs ===")
            print(runs_df.to_string(index=False))
            
        elif args.command == 'details':
            if not args.run_id:
                print("Error: --run-id required for details command")
                return
            get_run_details(db_manager, args.run_id)
            
        elif args.command == 'export':
            if not args.run_id:
                print("Error: --run-id required for export command")
                return
            export_results(db_manager, args.run_id, args.output)
            
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()