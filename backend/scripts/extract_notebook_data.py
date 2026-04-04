"""
Extract cleaned data from Kaggle notebook and save as CSV for training.
"""
import json
import pandas as pd
import sys
from pathlib import Path

def extract_data_from_notebook(notebook_path):
    """Extract dataframe from notebook cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Look for cells with variable assignments
    data_cells = []
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            # Look for dataframe assignments (df, data, cleaned, etc.)
            if 'df' in source.lower() or 'cleaned' in source.lower():
                data_cells.append(source)
    
    print(f"Found {len(data_cells)} potential data cells")
    return data_cells

def main():
    notebook_path = "c:\\Users\\Decade\\OneDrive\\Documents\\van lang\\AI\\vietnam-real-estate-catalyst-project-eda.ipynb"
    
    if not Path(notebook_path).exists():
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    print(f"📖 Reading notebook from:\n   {notebook_path}")
    
    # For now, we'll manually create a mapping based on the notebook structure
    # In practice, you would run the notebook cells to get the actual data
    
    print("\n⚠️  To extract cleaned data from your notebook:")
    print("   1. Run your notebook completely in Jupyter/Colab")
    print("   2. Export the cleaned dataframe to CSV:")
    print("      df.to_csv('vietnam_housing_cleaned.csv', index=False)")
    print("   3. Copy the CSV to: backend/data/vietnam_housing_cleaned.csv")
    print("   4. Run import_dataset.py to load into MongoDB")

if __name__ == "__main__":
    main()
