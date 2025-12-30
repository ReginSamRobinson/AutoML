"""Example script for running the AutoML engine."""

from __future__ import annotations

import warnings
from automl import run_dashboard

# Suppress sklearn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def main() -> None:
    """Launch the AutoML dashboard UI first, then run from there."""

    # Launch the dashboard without any results initially
    # The UI will provide upload and run capabilities
    print("\n" + "="*60)
    print("🚀 Launching AutoML Dashboard...")
    print("="*60)
    print("\n📊 UI is now ready!")
    print("   1. Upload your dataset")
    print("   2. Configure AutoML parameters")
    print("   3. Click 'Run AutoML' to start the engine")
    print("   4. View results in the dashboard\n")
    
    # Launch the dashboard with no initial results
    # UI starts in upload/config mode
    run_dashboard()


if __name__ == "__main__":
    main()
