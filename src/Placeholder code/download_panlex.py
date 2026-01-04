# download_panlex.py - Get PanLex dictionary data

import requests
from pathlib import Path
import gzip
import shutil


def download_panlex():
    """Download PanLex lexical database"""

    panlex_dir = Path("../../data")
    panlex_dir.mkdir(exist_ok=True)
    panlex_path = panlex_dir / "panlex.csv"

    if panlex_path.exists():
        print(f"✓ PanLex already exists at {panlex_path}")
        return

    print("Downloading PanLex data...")

    # Option 1: Use professor's panlex.csv if available
    # Copy from professor's directory or download from shared location

    # Option 2: Create minimal test data for Hindi
    # This is a tiny subset just to test your system
    test_data = """vocab\t639-3\tenglish name/var
flipkart\teng\tEnglish
member\teng\tEnglish
offer\teng\tEnglish
sath\thin\tHindi
saath\thin\tHindi
sat\thin\tHindi
bech\thin\tHindi
bik\thin\tHindi
raha\thin\tHindi
rha\thin\tHindi
hai\thin\tHindi
h\thin\tHindi
ka\thin\tHindi
ke\thin\tHindi
ye\thin\tHindi
yeh\thin\tHindi
lenovo\teng\tEnglish
smartphone\teng\tEnglish
shandar\thin\tHindi
shaandar\thin\tHindi
sandar\thin\tHindi
amazing\teng\tEnglish
wonderful\teng\tEnglish
selling\teng\tEnglish
with\teng\tEnglish
"""

    panlex_path.write_text(test_data, encoding="utf-8")
    print(f"✓ Created test PanLex at {panlex_path}")
    print(f"  → Added {len(test_data.splitlines())} entries")
    print("\n⚠️  This is a MINIMAL test dataset for development")
    print("   For production, you need the FULL PanLex database")
    print("   Ask your professor for: panlex.csv (full version)")


if __name__ == "__main__":
    download_panlex()
