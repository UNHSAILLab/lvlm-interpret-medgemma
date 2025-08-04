"""Main script to run one-pixel attack robustness evaluation."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np

from config import AttackConfig, ModelConfig, ExperimentConfig
from one_pixel_attack_helper import create_one_pixel_attacks, preprocess_images_only
from medgemma_evaluator import MedGemmaEvaluator
from database import DatabaseManager


def analyze_robustness(baseline_results: pd.DataFrame, attack_results: pd.DataFrame, preprocessed_results: pd.DataFrame = None) -> Dict:
    """Analyze robustness by comparing baseline, preprocessed, and attack results."""
    
    if preprocessed_results is None:
        # Original two-way comparison
        merged = pd.merge(
            baseline_results,
            attack_results,
            on='study_id',
            suffixes=('_baseline', '_attack')
        )
    else:
        # Three-way comparison
        # First rename columns to avoid conflicts
        baseline_renamed = baseline_results.rename(columns={
            col: f"{col}_baseline" if col != 'study_id' else col 
            for col in baseline_results.columns
        })
        preprocessed_renamed = preprocessed_results.rename(columns={
            col: f"{col}_preprocessed" if col != 'study_id' else col 
            for col in preprocessed_results.columns
        })
        attack_renamed = attack_results.rename(columns={
            col: f"{col}_attack" if col != 'study_id' else col 
            for col in attack_results.columns
        })
        
        # Now merge all three
        merged = pd.merge(baseline_renamed, preprocessed_renamed, on='study_id')
        merged = pd.merge(merged, attack_renamed, on='study_id')
    
    # Calculate metrics
    total_samples = len(merged)
    
    # Accuracy
    baseline_correct = (merged['model_answer_baseline'] == merged['correct_answer_baseline']).sum()
    attack_correct = (merged['model_answer_attack'] == merged['correct_answer_attack']).sum()
    
    baseline_accuracy = baseline_correct / total_samples
    attack_accuracy = attack_correct / total_samples
    
    # Attack success rate (changed predictions)
    changed_predictions = (merged['model_answer_baseline'] != merged['model_answer_attack']).sum()
    attack_success_rate = changed_predictions / total_samples
    
    # Targeted attack success (correct → incorrect)
    correct_to_incorrect = (
        (merged['model_answer_baseline'] == merged['correct_answer_baseline']) &
        (merged['model_answer_attack'] != merged['correct_answer_attack'])
    ).sum()
    
    targeted_success_rate = correct_to_incorrect / baseline_correct if baseline_correct > 0 else 0
    
    # Confidence analysis
    avg_confidence_baseline = merged['model_confidence_baseline'].mean()
    avg_confidence_attack = merged['model_confidence_attack'].mean()
    confidence_drop = avg_confidence_baseline - avg_confidence_attack
    
    # Initialize summary dict
    summary = {
        'total_samples': total_samples,
        'baseline_accuracy': baseline_accuracy,
        'attack_accuracy': attack_accuracy,
        'accuracy_drop': baseline_accuracy - attack_accuracy,
        'attack_success_rate': attack_success_rate,
        'targeted_success_rate': targeted_success_rate,
        'avg_confidence_baseline': avg_confidence_baseline,
        'avg_confidence_attack': avg_confidence_attack,
        'confidence_drop': confidence_drop,
    }
    
    # Add preprocessed metrics if available
    if preprocessed_results is not None:
        preprocessed_correct = (merged['model_answer_preprocessed'] == merged['correct_answer_preprocessed']).sum()
        preprocessed_accuracy = preprocessed_correct / total_samples
        avg_confidence_preprocessed = merged['model_confidence_preprocessed'].mean()
        
        summary.update({
            'preprocessed_accuracy': preprocessed_accuracy,
            'avg_confidence_preprocessed': avg_confidence_preprocessed,
            'preprocessing_effect': preprocessed_accuracy - baseline_accuracy,
            'attack_effect': attack_accuracy - preprocessed_accuracy,
        })
    
    # Per-question analysis
    if preprocessed_results is None:
        question_analysis = merged.groupby('question_baseline').agg({
            'model_answer_baseline': lambda x: (x == merged.loc[x.index, 'correct_answer_baseline']).mean(),
            'model_answer_attack': lambda x: (x == merged.loc[x.index, 'correct_answer_attack']).mean(),
        }).rename(columns={
            'model_answer_baseline': 'baseline_accuracy',
            'model_answer_attack': 'attack_accuracy'
        })
        question_analysis['accuracy_drop'] = question_analysis['baseline_accuracy'] - question_analysis['attack_accuracy']
    else:
        question_analysis = merged.groupby('question_baseline').agg({
            'model_answer_baseline': lambda x: (x == merged.loc[x.index, 'correct_answer_baseline']).mean(),
            'model_answer_preprocessed': lambda x: (x == merged.loc[x.index, 'correct_answer_preprocessed']).mean(),
            'model_answer_attack': lambda x: (x == merged.loc[x.index, 'correct_answer_attack']).mean(),
        }).rename(columns={
            'model_answer_baseline': 'baseline_accuracy',
            'model_answer_preprocessed': 'preprocessed_accuracy',
            'model_answer_attack': 'attack_accuracy'
        })
        question_analysis['preprocessing_effect'] = question_analysis['preprocessed_accuracy'] - question_analysis['baseline_accuracy']
        question_analysis['attack_effect'] = question_analysis['attack_accuracy'] - question_analysis['preprocessed_accuracy']
        question_analysis['total_effect'] = question_analysis['attack_accuracy'] - question_analysis['baseline_accuracy']
    
    return {
        'summary': summary,
        'per_question': question_analysis,
        'detailed_results': merged
    }


def main(args):
    """Main function to run the complete evaluation pipeline."""
    
    # Setup configuration
    attack_config = AttackConfig(
        source_folder=Path(args.source_folder),
        output_folder=Path(args.output_folder),
        max_iter=args.max_iter,
        pop_size=args.pop_size,
        pathology=args.pathology
    )
    
    model_config = ModelConfig()
    
    experiment_config = ExperimentConfig(
        attack=attack_config,
        model=model_config,
        data_path=Path(args.data_path),
        results_folder=Path(args.results_folder)
    )
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_folder = experiment_config.results_folder / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize database manager
    db_manager = DatabaseManager() if args.use_database else None
    
    # Load data
    print(f"Loading data from {experiment_config.data_path}")
    data = pd.read_csv(experiment_config.data_path)
    
    if args.limit:
        print(f"Limiting to {args.limit} samples")
        data = data.head(args.limit)
    
    # Insert questions into database if using database
    if db_manager:
        print("Inserting questions into database...")
        inserted = db_manager.insert_questions(data)
        print(f"Inserted {inserted} new questions into database")
    
    # Step 1: Preprocess images without attacks
    preprocessed_folder = Path(args.preprocessed_folder) if args.preprocessed_folder else Path(str(attack_config.output_folder).replace('onepix', 'preprocessed'))
    if not args.skip_preprocessing:
        print("\n=== Step 1: Preprocessing Images (No Attack) ===")
        preprocess_results = preprocess_images_only(
            df=data,
            source_folder=str(attack_config.source_folder),
            output_folder=str(preprocessed_folder),
            run_id=run_id,
            db_manager=db_manager
        )
        preprocess_summary_path = run_folder / 'preprocessing_summary.csv'
        preprocess_results.to_csv(preprocess_summary_path, index=False)
        print(f"Preprocessing summary saved to {preprocess_summary_path}")
    else:
        print("Skipping preprocessing (--skip-preprocessing flag set)")
    
    # Step 2: Generate one-pixel attacks
    if not args.skip_attack_generation:
        print("\n=== Step 2: Generating One-Pixel Attacks ===")
        attack_results = create_one_pixel_attacks(
            df=data,
            source_folder=str(attack_config.source_folder),
            output_folder=str(attack_config.output_folder),
            pathology=attack_config.pathology,
            max_iter=attack_config.max_iter,
            pop_size=attack_config.pop_size,
            log_file=f'attack_log_{timestamp}.csv',
            run_id=run_id,
            db_manager=db_manager
        )
        attack_summary_path = run_folder / 'attack_generation_summary.csv'
        attack_results.to_csv(attack_summary_path, index=False)
        print(f"Attack generation summary saved to {attack_summary_path}")
    else:
        print("Skipping attack generation (--skip-attack-generation flag set)")
    
    # Step 3: Initialize MedGemma evaluator
    print("\n=== Step 3: Initializing MedGemma Model ===")
    evaluator = MedGemmaEvaluator(model_config)
    
    # Step 4: Run baseline evaluation
    print("\n=== Step 4: Running Baseline Evaluation ===")
    baseline_results_path = run_folder / 'baseline_results.csv'
    baseline_results = evaluator.evaluate_dataset(
        data=data,
        folder_path=attack_config.source_folder,
        attack_type='baseline',
        save_path=baseline_results_path,
        resume_from=args.resume_baseline,
        run_id=run_id,
        db_manager=db_manager
    )
    
    # Step 5: Run preprocessed evaluation
    preprocessed_eval_results = None
    if args.include_preprocessed_eval:
        print("\n=== Step 5: Running Preprocessed (No Attack) Evaluation ===")
        preprocessed_results_path = run_folder / 'preprocessed_results.csv'
        preprocessed_eval_results = evaluator.evaluate_dataset(
            data=data,
            folder_path=preprocessed_folder,
            attack_type='preprocessed',
            save_path=preprocessed_results_path,
            resume_from=args.resume_preprocessed,
            run_id=run_id,
            db_manager=db_manager
        )
    
    # Step 6: Run attack evaluation
    print(f"\n=== Step {'6' if args.include_preprocessed_eval else '5'}: Running One-Pixel Attack Evaluation ===")
    attack_results_path = run_folder / 'onepixel_results.csv'
    attack_eval_results = evaluator.evaluate_dataset(
        data=data,
        folder_path=attack_config.output_folder,
        attack_type='onepixel',
        save_path=attack_results_path,
        resume_from=args.resume_attack,
        run_id=run_id,
        db_manager=db_manager
    )
    
    # Step 7: Analyze robustness
    print(f"\n=== Step {'7' if args.include_preprocessed_eval else '6'}: Analyzing Robustness ===")
    robustness_analysis = analyze_robustness(baseline_results, attack_eval_results, preprocessed_eval_results)
    
    # Save robustness analysis
    summary_path = run_folder / 'robustness_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(robustness_analysis['summary'], f, indent=2)
    
    # Save per-question analysis
    question_analysis_path = run_folder / 'per_question_analysis.csv'
    robustness_analysis['per_question'].to_csv(question_analysis_path)
    
    # Save detailed comparison
    detailed_path = run_folder / 'detailed_comparison.csv'
    robustness_analysis['detailed_results'].to_csv(detailed_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*60)
    for key, value in robustness_analysis['summary'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)
    print(f"\nAll results saved to: {run_folder}")
    
    # Get database summary if using database
    if db_manager:
        print("\n=== Database Summary ===")
        db_summary = db_manager.get_robustness_summary(run_id)
        for key, value in db_summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        db_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-pixel attack robustness evaluation")
    
    # Data paths
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to CSV file with questions and image paths')
    parser.add_argument('--source-folder', type=str, required=True,
                        help='Root folder containing original images')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Folder to save adversarial images')
    parser.add_argument('--results-folder', type=str, default='./results',
                        help='Folder to save evaluation results')
    
    # Attack parameters
    parser.add_argument('--pathology', type=str, default='Pneumonia',
                        help='Target pathology for attacks')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum iterations for differential evolution')
    parser.add_argument('--pop-size', type=int, default=200,
                        help='Population size for differential evolution')
    
    # Execution options
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process')
    parser.add_argument('--skip-attack-generation', action='store_true',
                        help='Skip attack generation if already done')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing step if already done')
    parser.add_argument('--include-preprocessed-eval', action='store_true',
                        help='Include evaluation of preprocessed images without attacks')
    parser.add_argument('--preprocessed-folder', type=str, default=None,
                        help='Folder containing preprocessed images (auto-generated if not specified)')
    parser.add_argument('--resume-baseline', type=str, default=None,
                        help='Resume baseline evaluation from CSV file')
    parser.add_argument('--resume-preprocessed', type=str, default=None,
                        help='Resume preprocessed evaluation from CSV file')
    parser.add_argument('--resume-attack', type=str, default=None,
                        help='Resume attack evaluation from CSV file')
    
    # Database options
    parser.add_argument('--use-database', action='store_true',
                        help='Store results in PostgreSQL database')
    parser.add_argument('--database-url', type=str, default=None,
                        help='PostgreSQL connection string (uses env var DATABASE_URL if not provided)')
    
    args = parser.parse_args()
    main(args)