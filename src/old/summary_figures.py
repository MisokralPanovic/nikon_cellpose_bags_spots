import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Dict

def create_summary_figures(results: List[Dict], config: Dict):
    """Create summary analysis figures.

    Args:
        results (List[Dict]): List of ROI result dictionaries.
        config (Dict): Configuration with paths.
    """
    df = pd.DataFrame(results)
    figures_folder = config['paths']['figures']
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    create_summary_boxplot(df, figures_folder)
    create_size_correlation_figure(df, figures_folder)

def create_summary_boxplot(df: pd.DataFrame, output_folder: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Spot count distribution
    sns.boxplot(data=df, y='Condition', x='Spot_Count', ax=axes[0, 0])
    axes[0, 0].set_title('Spot Count Distribution by Condition')
    axes[0, 0].set_ylabel('')
    add_n_annotations(axes[0, 0], df)
    
    # ROI area distribution
    sns.boxplot(data=df, y='Condition', x='ROI_Area_um2', ax=axes[0, 1])
    axes[0, 1].set_title('ROI Area Distribution by Condition')
    axes[0, 1].set_xlabel('ROI Area (μm²)')
    axes[0, 1].set_ylabel('')
    add_n_annotations(axes[0, 1], df)
    
    # Spots per area
    sns.boxplot(data=df, y='Condition', x='Spots_per_Area', ax=axes[1, 0])
    axes[1, 0].set_title('Spots per Area by Condition')
    axes[1, 0].set_xlabel('Spots per μm²')
    axes[1, 0].set_ylabel('')
    add_n_annotations(axes[1, 0], df)
    
    # Spots per volume
    sns.boxplot(data=df, y='Condition', x='Spots_per_Volume', ax=axes[1, 1])
    axes[1, 1].set_title('Spots per Volume by Condition')
    axes[1, 1].set_xlabel('Spots per μm³')
    axes[1, 1].set_ylabel('')
    add_n_annotations(axes[1, 1], df)
    
    plt.tight_layout()
    summary_path = output_folder / 'Summary_Analysis.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_size_correlation_figure(df: pd.DataFrame, output_folder: Path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    sns.scatterplot(data=df, x='ROI_Area_um2', y='Spot_Count', hue='Condition', ax=ax)
    ax.set_title('Spot Count vs ROI Area')
    ax.set_xlabel('ROI Area (μm²)')
    ax.set_ylabel('Spot Count')
    
    plt.tight_layout()
    correlation_path = output_folder / 'Correlation_Analysis.png'
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
    plt.close(fig)   

def add_n_annotations(ax, df: pd.DataFrame):
    conditions = df['Condition'].unique()
    
    for i, condition in enumerate(conditions):
        n = len(df[df['Condition'] == condition])
        ax.text(ax.get_xlim()[1] * 0.98, i, f"n={n}",
                verticalalignment='center', 
                horizontalalignment='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))