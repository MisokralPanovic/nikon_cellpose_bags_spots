from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def save_results(results: List[Dict], config: Dict):
    """Save results to CSV and Excel files.
    
    Args:
        results (List[Dict]): List of ROI result dictionaries.
        config (Dict): Configuration with paths.  
    """
    df = pd.DataFrame(results)
    
    experiment_name = config['experiment']['name']
    output_folder = config['paths']['processed_data']
    
    csv_path = output_folder / f"{experiment_name}_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved .csv to: {csv_path}")
    
    excel_path = output_folder / f"{experiment_name}_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Data', index=False)
        
        summary = df.groupby('Condition').agg({
            'Spot_Count': ['count', 'mean', 'std', 'sum'],
            'ROI_Area_um2': ['mean', 'std'],
            'Spots_per_Area': ['mean', 'std'],
            'Spots_per_Volume': ['mean', 'std']
        }).round(3)       
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary.reset_index().to_excel(writer, sheet_name='Summary_by_Condition', index=False)
        
    logger.info(f"Saved .xlsx to: {excel_path}")

    # log statistics
    logger.info(f"Total ROIs: {len(df)}")
    logger.info(f"Conditions: {df['Condition'].nunique()}")
    logger.info(f"Total spots: {df['Spot_Count'].sum()}")
    logger.info(f"Average spots per ROI: {df['Spot_Count'].mean():.1f} ± {df['Spot_Count'].std():.1f}")
