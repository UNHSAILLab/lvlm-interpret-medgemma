"""General pathology ground truth extractor for VQA datasets."""

def extract_pathology_ground_truth(row, target_pathology='Pneumonia'):
    """Extract binary pathology presence from VQA answer.
    
    This is a more general version that works for any pathology,
    not just pneumonia.
    
    Args:
        row: DataFrame row containing question and answer information
        target_pathology: The pathology to check for (default: 'Pneumonia')
        
    Returns:
        int: 1 if pathology is present, 0 otherwise
    """
    question = row['question'].lower()
    correct_answer = str(row['correct_answer']).lower().strip()
    pathology_lower = target_pathology.lower()
    
    # Handle direct yes/no answers (MIMIC format)
    if correct_answer in ['yes', 'no']:
        # For questions about the specific pathology
        if pathology_lower in question:
            return 1 if correct_answer == 'yes' else 0
        # For questions about effusion when target is Effusion
        elif pathology_lower == 'effusion' and 'effusion' in question:
            return 1 if correct_answer == 'yes' else 0
        # For general pathology presence questions
        elif any(term in question for term in ['evidence of', 'is there', 'presence']):
            return 1 if correct_answer == 'yes' else 0
        else:
            # Default: yes means pathology present
            return 1 if correct_answer == 'yes' else 0
    
    # Handle A/B/C/D format (original format)
    elif len(correct_answer) == 1 and correct_answer in 'abcd':
        try:
            answer_idx = ord(correct_answer.upper()) - ord('A')
            if isinstance(row['options'], list) and answer_idx < len(row['options']):
                answer_text = str(row['options'][answer_idx]).lower()
                
                # Check for pathology presence/absence
                if 'absence' in answer_text or f'no {pathology_lower}' in answer_text:
                    return 0
                elif 'presence' in answer_text or pathology_lower in answer_text:
                    return 1
                elif answer_text in ['yes', 'no']:
                    return 1 if answer_text == 'yes' else 0
        except:
            pass
    
    # Default based on answer
    return 1 if correct_answer == 'yes' else 0