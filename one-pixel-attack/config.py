"""Configuration for one-pixel attack experiments."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class AttackConfig:
    """Configuration for one-pixel attacks."""
    source_folder: Path
    output_folder: Path
    max_iter: int = 100
    pop_size: int = 200
    pathology: str = 'Pneumonia'
    log_file: str = 'one_pixel_attack_log.csv'
    
    def __post_init__(self):
        self.source_folder = Path(self.source_folder)
        self.output_folder = Path(self.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for MedGemma model."""
    model_id: str = 'google/medgemma-4b-it'
    device_map: str = 'auto'
    torch_dtype: str = 'bfloat16'
    max_new_tokens: int = 150
    do_sample: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    attack: AttackConfig
    model: ModelConfig
    data_path: Path
    results_folder: Path
    
    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.results_folder = Path(self.results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)