"""Utilities for handling dataset uploads in the AutoML dashboard."""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_upload_folder(base_path: Optional[str] = None) -> str:
    """Create uploads folder if it doesn't exist."""
    if base_path is None:
        base_path = Path(__file__).parent.parent / 'uploads'
    else:
        base_path = Path(base_path) / 'uploads'
    
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


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
    import numpy as np
    import json
    
    try:
        numeric_summary = df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
    except:
        numeric_summary = {}
    
    # Convert dtypes to JSON-serializable strings
    dtypes_dict = {}
    for col, dtype in df.dtypes.to_dict().items():
        dtypes_dict[str(col)] = str(dtype)
    
    # Convert missing values to JSON-serializable format
    missing_values = {}
    for col, count in df.isnull().sum().to_dict().items():
        missing_values[str(col)] = int(count)
    
    # Replace NaN and infinity with None for JSON serialization
    def clean_for_json(obj):
        """Recursively clean objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return obj
    
    head_data = df.head(rows).to_dict(orient='records')
    head_data = clean_for_json(head_data)
    
    numeric_summary = clean_for_json(numeric_summary)
    
    return {
        'shape': list(df.shape),  # Convert tuple to list for JSON serialization
        'columns': [str(c) for c in df.columns.tolist()],
        'dtypes': dtypes_dict,
        'head': head_data,
        'missing_values': missing_values,
        'numeric_summary': numeric_summary,
    }


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
