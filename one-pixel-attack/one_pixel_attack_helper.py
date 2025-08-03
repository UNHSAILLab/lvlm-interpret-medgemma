"""One-pixel adversarial attack implementation for medical X-ray images.

This module provides functionality to perform one-pixel adversarial attacks
on chest X-ray images, specifically targeting pneumonia detection models.
The attacks are constrained to lung regions for medical relevance.
"""

# Standard library imports
import os
import ast
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.optimize import differential_evolution
from skimage import morphology, measure, filters
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
import torchxrayvision as xrv

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# Constants
DEFAULT_MAX_ITER = 100
DEFAULT_POP_SIZE = 200
IMAGE_SIZE = 224
MIN_OBJECT_SIZE = 1000
HOLE_THRESHOLD = 500
DISK_RADIUS = 5
BORDER_THRESHOLD = 10
GAUSSIAN_SIGMA = 2
DEFAULT_ALPHA = 0.3

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res_model = xrv.models.DenseNet(weights="densenet121-res224-all")
res_model.eval()
res_model = res_model.to(device)

PATHOLOGY_MAP = {pathology: idx for idx, pathology in enumerate(res_model.pathologies)}

# ==================== Data Processing Functions ====================

def extract_pneumonia_ground_truth(row: pd.Series) -> int:
    """Extract binary pneumonia presence from VQA answer.
    
    Args:
        row: DataFrame row containing question and answer information
        
    Returns:
        int: 1 if pneumonia is present, 0 otherwise
    """
    question = row['question'].lower()
    answer_text = row['options'][ord(row['correct_answer']) - ord('A')].lower()

    # Check if this is a presence/absence question
    if 'absence' in question or 'present' in question or 'finding regarding' in question:
        if 'absence' in answer_text or 'no pneumonia' in answer_text:
            return 0  # No pneumonia
        elif 'presence' in answer_text or 'pneumonia' in answer_text:
            return 1  # Pneumonia present

    # For location questions, pneumonia is implicitly present
    if 'location' in question or 'where' in question:
        return 1  # Pneumonia must be present if asking about location

    # Default: assume pneumonia present (since filtered for pneumonia questions)
    return 1


# ==================== Image Processing Functions ====================

def get_lung_mask_adaptive(image_224: np.ndarray) -> np.ndarray:
    """Extract lung regions using adaptive thresholding.
    
    Args:
        image_224: Grayscale image array of size 224x224
        
    Returns:
        np.ndarray: Binary mask where True indicates lung regions
    """
    # Detect and mask out black border
    border_mask = image_224 > BORDER_THRESHOLD

    # Apply Gaussian filter to smooth
    smoothed = gaussian(image_224, sigma=GAUSSIAN_SIGMA)

    # Use Otsu's method only on non-border pixels
    non_border_pixels = smoothed[border_mask]
    if len(non_border_pixels) > 0:
        threshold = threshold_otsu(non_border_pixels)
    else:
        threshold = 100

    # Create binary mask
    binary = (smoothed < threshold) & border_mask

    # Morphological operations
    binary = morphology.binary_closing(binary, morphology.disk(DISK_RADIUS))
    binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)
    binary = morphology.remove_small_holes(binary, area_threshold=HOLE_THRESHOLD)

    # Select largest components
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    regions_sorted = sorted(regions, key=lambda x: x.area, reverse=True)

    lung_mask = np.zeros_like(binary)
    for region in regions_sorted[:2]:
        lung_mask[labeled == region.label] = True

    return lung_mask


def overlay_lung_mask(image_224: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Create RGB image with red lung overlay for visualization.
    
    Args:
        image_224: Grayscale image array of size 224x224
        alpha: Transparency factor for overlay (0-1)
        
    Returns:
        np.ndarray: RGB image with lung regions highlighted in red
    """
    mask = get_lung_mask_adaptive(image_224)
    img = image_224.astype(float)
    if img.max() > 1:
        img /= 255
    rgb = np.stack([img]*3, axis=-1)
    red_overlay = rgb.copy()
    red_overlay[..., 0][mask] = 1
    red_overlay[..., 1][mask] = 0
    red_overlay[..., 2][mask] = 0
    return (1 - alpha) * rgb + alpha * red_overlay


# ==================== Attack Implementation ====================

class OnePixelAttackLungConstrained:
    """One-pixel adversarial attack constrained to lung regions.
    
    This class implements adversarial attacks that modify a single pixel
    within detected lung regions to fool pneumonia detection models.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """Initialize the attacker with a model.
        
        Args:
            model: Pre-trained PyTorch model for chest X-ray classification
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.lung_mask = None
        self.valid_coords = None

    def set_lung_mask(self, mask: np.ndarray) -> None:
        """Set the lung region mask for constraining attacks.
        
        Args:
            mask: Binary mask indicating valid lung regions
        """
        self.lung_mask = mask
        self.valid_coords = np.argwhere(mask)

    def perturb_image(self, image: np.ndarray, x: int, y: int, delta: float) -> np.ndarray:
        """Apply single-pixel perturbation to an image.
        
        Args:
            image: Original image array
            x: X-coordinate of pixel to modify
            y: Y-coordinate of pixel to modify
            delta: Perturbation magnitude
            
        Returns:
            np.ndarray: Perturbed image
        """
        perturbed = image.copy()
        x, y = int(x), int(y)
        perturbed[y, x] = np.clip(perturbed[y, x] + delta, 0, 255)
        return perturbed

    def preprocess_for_model(self, image_224: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image_224: Image array of size 224x224
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model input
        """
        image_norm = xrv.datasets.normalize(image_224, maxval=255)
        image_tensor = torch.from_numpy(image_norm).float()
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor.to(self.device)

    def fitness_function(self, params: np.ndarray, image: np.ndarray, 
                        true_label: int, pathology_idx: int) -> float:
        """Fitness function for optimization algorithm.
        
        Args:
            params: [x, y, delta] parameters for the attack
            image: Original image
            true_label: Ground truth label (0 or 1)
            pathology_idx: Index of the pathology to target
            
        Returns:
            float: Negative fitness value (to minimize)
        """
        x, y, delta = params
        x, y = int(x), int(y)

        # Check if pixel is in lung region
        if self.lung_mask is not None and not self.lung_mask[y, x]:
            return 1e6  # Large penalty for pixels outside lungs

        perturbed_image = self.perturb_image(image, x, y, delta)
        image_tensor = self.preprocess_for_model(perturbed_image)

        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]

        prob = probs[pathology_idx]

        if true_label == 1:
            fitness = 1 - prob
        else:
            fitness = prob

        return -fitness


# ==================== Helper Functions ====================

def _initialize_log_file(log_path: str) -> None:
    """Initialize CSV log file with headers if it doesn't exist."""
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("timestamp,study_id,image_path,attack_type,original_prob,adversarial_prob,"
                    "pixel_x_224,pixel_y_224,pixel_x_orig,pixel_y_orig,"
                    "original_pixel_value,new_pixel_value,pixel_delta,"
                    "in_lung_mask,lung_pixels_count,total_pixels,"
                    "success,prob_change,iterations,fitness\n")


def _load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and preprocess an image, handling different formats.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (8-bit image array, original shape)
    """
    img = Image.open(image_path)
    
    # Handle different image modes
    if img.mode == 'I;16':
        img_array = np.array(img)
        img_min, img_max = img_array.min(), img_array.max()
        img_8bit = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_8bit = np.array(img.convert('L'))
    
    return img_8bit, img_8bit.shape


def _scale_coordinates(x_224: int, y_224: int, 
                      original_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Scale coordinates from 224x224 to original image size.
    
    Args:
        x_224: X-coordinate in 224x224 space
        y_224: Y-coordinate in 224x224 space
        original_shape: Original image shape (height, width)
        
    Returns:
        Tuple of (x_orig, y_orig) in original image space
    """
    scale_x = original_shape[1] / IMAGE_SIZE
    scale_y = original_shape[0] / IMAGE_SIZE
    return int(x_224 * scale_x), int(y_224 * scale_y)


def _write_log_entry(log_path: str, entry: Dict) -> None:
    """Append a log entry to the CSV file."""
    with open(log_path, 'a') as f:
        f.write(f"{entry['timestamp']},{entry['study_id']},{entry['image_path']},"
               f"{entry['attack_type']},{entry['original_prob']:.6f},"
               f"{entry['adversarial_prob']:.6f},{entry['pixel_x_224']},"
               f"{entry['pixel_y_224']},{entry['pixel_x_orig']},"
               f"{entry['pixel_y_orig']},{entry['original_pixel_value']},"
               f"{entry['new_pixel_value']},{entry['pixel_delta']:.2f},"
               f"{entry['in_lung_mask']},{entry['lung_pixels_count']},"
               f"{entry['total_pixels']},{entry['success']},"
               f"{entry['prob_change']:.6f},{entry['iterations']},"
               f"{entry['fitness']:.6f}\n")


# ==================== Main Attack Function ====================

def create_one_pixel_attacks(
    df: pd.DataFrame,
    source_folder: str,
    output_folder: str = 'deid_png_onepix',
    pathology: str = 'Pneumonia',
    max_iter: int = DEFAULT_MAX_ITER,
    pop_size: int = DEFAULT_POP_SIZE,
    log_file: str = 'one_pixel_attack_log.csv',
    run_id: Optional[str] = None,
    db_manager: Optional['DatabaseManager'] = None
) -> pd.DataFrame:
    """Create one-pixel adversarial attacks for X-ray images.
    
    This function generates adversarial examples by modifying a single pixel
    within lung regions to fool pneumonia detection models.
    
    Args:
        df: DataFrame containing image paths and metadata
        source_folder: Root folder containing source images
        output_folder: Folder to save adversarial images
        pathology: Target pathology for the attack
        max_iter: Maximum iterations for optimization
        pop_size: Population size for differential evolution
        log_file: Name of the CSV log file
        
    Returns:
        pd.DataFrame: Summary of attack results
    """

    os.makedirs(output_folder, exist_ok=True)
    pathology_idx = PATHOLOGY_MAP[pathology]
    attacker = OnePixelAttackLungConstrained(res_model, device)
    print("Model Loaded .......!")

    attack_results = []

    # Initialize log file
    log_path = os.path.join(output_folder, log_file)
    _initialize_log_file(log_path)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating one-pixel attacks"):
        try:
            # Parse image paths
            if isinstance(row['image_path'], str):
                image_paths = ast.literal_eval(row['image_path'])
            else:
                image_paths = row['image_path']

            for img_path in image_paths:
                # Clean path
                relative_path = img_path.replace('../', '')
                full_input_path = os.path.join(source_folder, relative_path)

                if not os.path.exists(full_input_path):
                    print(f"✗ File not found: {full_input_path}")
                    continue

                # Create output path
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Skip if already exists
                if os.path.exists(output_path):
                    print(f"Skipping existing: {output_path}")
                    continue

                # Load and preprocess image
                original_image, original_shape = _load_and_preprocess_image(full_input_path)

                # Create preprocessed version for finding pixel location
                image_224 = resize(original_image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
                image_224 = (image_224 * 255).astype(np.uint8)

                # Get lung mask on preprocessed image
                lung_mask = get_lung_mask_adaptive(image_224)
                lung_pixels_count = np.sum(lung_mask)
                attacker.set_lung_mask(lung_mask)

                # Get original prediction using preprocessed original image
                with torch.no_grad():
                    orig_tensor = attacker.preprocess_for_model(image_224)
                    orig_output = res_model(orig_tensor)
                    orig_prob = torch.sigmoid(orig_output)[0, pathology_idx].cpu().item()

                # Determine target
                true_label = extract_pneumonia_ground_truth(row)

                # Modified fitness function that applies perturbation to original image
                def fitness_with_original(params):
                    x_224, y_224, delta = params
                    x_224, y_224 = int(x_224), int(y_224)

                    # Check if pixel is in lung region (on 224x224 mask)
                    if lung_mask is not None and not lung_mask[y_224, x_224]:
                        return 1e6

                    # Scale coordinates to original size
                    orig_x, orig_y = _scale_coordinates(x_224, y_224, original_shape)

                    # Apply perturbation to original image
                    perturbed_original = original_image.copy()
                    perturbed_original[orig_y, orig_x] = np.clip(
                        perturbed_original[orig_y, orig_x] + delta, 0, 255
                    )

                    # Preprocess the perturbed original image
                    perturbed_224 = resize(perturbed_original, (224, 224), anti_aliasing=True)
                    perturbed_224 = (perturbed_224 * 255).astype(np.uint8)

                    # Get prediction
                    image_tensor = attacker.preprocess_for_model(perturbed_224)
                    with torch.no_grad():
                        output = res_model(image_tensor)
                        prob = torch.sigmoid(output)[0, pathology_idx].cpu().item()

                    # Calculate fitness
                    if true_label == 1:
                        fitness = 1 - prob
                    else:
                        fitness = prob

                    return -fitness

                # Run attack
                bounds = [(0, 223), (0, 223), (-255, 255)]

                result = differential_evolution(
                    func=fitness_with_original,
                    bounds=bounds,
                    maxiter=max_iter,
                    popsize=pop_size,
                    seed=42,
                    workers=1
                )

                # Get best pixel location and delta
                best_x_224, best_y_224 = int(result.x[0]), int(result.x[1])
                best_delta = result.x[2]

                # Scale to original coordinates
                best_x_orig, best_y_orig = _scale_coordinates(best_x_224, best_y_224, original_shape)

                # Get original pixel value
                original_pixel_value = original_image[best_y_orig, best_x_orig]

                # Create adversarial image by modifying original
                adv_image_full = original_image.copy()
                adv_image_full[best_y_orig, best_x_orig] = np.clip(
                    adv_image_full[best_y_orig, best_x_orig] + best_delta, 0, 255
                )
                new_pixel_value = adv_image_full[best_y_orig, best_x_orig]

                # Save adversarial image
                Image.fromarray(adv_image_full).save(output_path)

                # Verify attack by preprocessing the adversarial original image
                adv_image_224 = resize(adv_image_full, (224, 224), anti_aliasing=True)
                adv_image_224 = (adv_image_224 * 255).astype(np.uint8)

                with torch.no_grad():
                    adv_tensor = attacker.preprocess_for_model(adv_image_224)
                    adv_output = res_model(adv_tensor)
                    adv_prob = torch.sigmoid(adv_output)[0, pathology_idx].cpu().item()

                # Check if pixel is in lung mask
                in_lung = lung_mask[best_y_224, best_x_224]

                # Calculate success
                success = (true_label == 1 and adv_prob < 0.5) or (true_label == 0 and adv_prob >= 0.5)
                prob_change = abs(adv_prob - orig_prob)

                # Detailed logging entry
                log_entry = {
                    'timestamp': pd.Timestamp.now(),
                    'study_id': row['study_id'],
                    'patient_id': row['PatientID'],
                    'ImagePath': row['ImagePath'],
                    'image_path': img_path,
                    'attack_type': 'one_pixel',
                    'original_prob': orig_prob,
                    'adversarial_prob': adv_prob,
                    'pixel_x_224': best_x_224,
                    'pixel_y_224': best_y_224,
                    'pixel_x_orig': best_x_orig,
                    'pixel_y_orig': best_y_orig,
                    'original_pixel_value': int(original_pixel_value),
                    'new_pixel_value': int(new_pixel_value),
                    'pixel_delta': best_delta,
                    'in_lung_mask': in_lung,
                    'lung_pixels_count': lung_pixels_count,
                    'total_pixels': IMAGE_SIZE * IMAGE_SIZE,
                    'success': success,
                    'prob_change': prob_change,
                    'iterations': result.nit,
                    'fitness': -result.fun
                }

                # Record results
                attack_results.append(log_entry)

                # Save to database if available
                if db_manager and run_id:
                    log_entry['run_id'] = run_id
                    db_manager.insert_attack_log(run_id, log_entry)

                # Append to CSV log file
                _write_log_entry(log_path, log_entry)

                print(f"✓ {row['study_id']}: {orig_prob:.3f} → {adv_prob:.3f} "
                      f"[Pixel: ({best_x_orig},{best_y_orig}), Δ={best_delta:.1f}, In lung: {in_lung}]")

        except Exception as e:
            print(f"✗ Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final DataFrame
    results_df = pd.DataFrame(attack_results)
    results_df.to_csv(os.path.join(output_folder, 'attack_summary.csv'), index=False)

    # Generate summary statistics
    if len(results_df) > 0:
        _print_attack_summary(results_df)

    return results_df


def _print_attack_summary(results_df: pd.DataFrame) -> None:
    """Print a summary of attack results."""
    print("\n" + "="*50)
    print("ATTACK SUMMARY")
    print("="*50)
    print(f"Total attacks: {len(results_df)}")
    print(f"Successful attacks: {results_df['success'].sum()} ({results_df['success'].mean()*100:.1f}%)")
    print(f"Average probability change: {results_df['prob_change'].mean():.4f}")
    print(f"Attacks in lung mask: {results_df['in_lung_mask'].sum()} ({results_df['in_lung_mask'].mean()*100:.1f}%)")
    print(f"Average pixel delta: {results_df['pixel_delta'].mean():.1f}")
    print(f"Average iterations: {results_df['iterations'].mean():.1f}")