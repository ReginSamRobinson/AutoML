"""Utilities for handling dataset uploads in the AutoML dashboard."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
SAMPLE_DATASET_FILENAME = "sample_breast_cancer.csv"


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_upload_path(folder: Path, filename: str) -> Path:
    """Return a path in folder that does not overwrite an existing file."""
    candidate = folder / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while candidate.exists():
        candidate = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def ensure_sample_dataset(upload_dir: Path) -> Optional[Path]:
    """Write the breast-cancer sample dataset to upload_dir if missing.

    Returns the sample path, or None if it could not be created.
    """
    sample_path = Path(upload_dir) / SAMPLE_DATASET_FILENAME
    if sample_path.exists():
        return sample_path

    try:
        from sklearn.datasets import load_breast_cancer

        load_breast_cancer(as_frame=True).frame.to_csv(sample_path, index=False)
    except Exception:
        sample_path.unlink(missing_ok=True)
        return None
    return sample_path


def create_upload_folder(base_path: Optional[str] = None) -> str:
    """Create uploads folder if it doesn't exist."""
    if base_path is None:
        base_path = Path(__file__).parent.parent / 'uploads'
    else:
        base_path = Path(base_path) / 'uploads'
    
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


def _coerce_json_value(value):
    if isinstance(value, dict):
        return {str(key): _coerce_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return getattr(value, "item")()
        except Exception:
            return value
    return value


def load_dataset(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load dataset from file.
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        file_ext = file_path.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'json':
            df = pd.read_json(file_path)
        elif file_ext == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            return None, f"Unsupported file format: {file_ext}"
        
        if df.empty:
            return None, "Dataset is empty"
        
        return df, None
    
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def get_dataset_preview(df: pd.DataFrame, rows: int = 5) -> dict:
    """Get preview of dataset with basic statistics."""
    preview = {
        'shape': tuple(int(x) for x in df.shape),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'head': df.head(rows).to_dict(orient='records'),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
    }
    return _coerce_json_value(preview)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate dataset for AutoML.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataset is empty"
    
    if df.shape[0] < 10:
        return False, "Dataset has fewer than 10 rows"
    
    # Check for at least 2 columns
    if df.shape[1] < 2:
        return False, "Dataset must have at least 2 columns"
    
    # Check if there are any non-null values
    if df.isnull().all().all():
        return False, "All values in dataset are null"
    
    return True, None
