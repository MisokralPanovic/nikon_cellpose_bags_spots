# Cell 9: Create summary analysis figures
if results:
    df = pd.DataFrame(results)
    # Filter out ROIs with area < 3000 µm²
    df = df[df['ROI_Area_um2'] >= 3000]
    
    # Calculate sample sizes for each condition
    sample_sizes = df.groupby('Condition').size()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Spot counts by condition
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Function to add sample size annotations
    def add_n_annotations(ax, data_col, group_col='Condition'):
        """Add sample size annotations to boxplot"""
        # Get unique conditions and their positions
        conditions = df[group_col].unique()
        
        for i, condition in enumerate(conditions):
            n = len(df[df[group_col] == condition])
            # Add annotation to the right of each boxplot
            ax.text(ax.get_xlim()[1] * 0.98, i, f'n={n}', 
                   verticalalignment='center', 
                   horizontalalignment='right',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Spot count distribution
    sns.boxplot(data=df, y='Condition', x='Spot_Count', ax=axes[0, 0])
    axes[0, 0].set_title('Spot Count Distribution by Condition')
    axes[0, 0].set_ylabel('')
    add_n_annotations(axes[0, 0], 'Spot_Count')
    
    # ROI area distribution
    sns.boxplot(data=df, y='Condition', x='ROI_Area_um2', ax=axes[0, 1])
    axes[0, 1].set_title('ROI Area Distribution by Condition')
    axes[0, 1].set_xlabel('ROI Area (μm²)')
    axes[0, 1].set_ylabel('')
    add_n_annotations(axes[0, 1], 'ROI_Area_um2')
    
    # Spots per area
    sns.boxplot(data=df, y='Condition', x='Spots_per_Area', ax=axes[1, 0])
    axes[1, 0].set_title('Spots per Area by Condition')
    axes[1, 0].set_xlabel('Spots per μm²')
    axes[1, 0].set_ylabel('')
    add_n_annotations(axes[1, 0], 'Spots_per_Area')
    
    # Spots per volume
    sns.boxplot(data=df, y='Condition', x='Spots_per_Volume', ax=axes[1, 1])
    axes[1, 1].set_title('Spots per Volume by Condition')
    axes[1, 1].set_xlabel('Spots per μm³')
    axes[1, 1].set_ylabel('')
    add_n_annotations(axes[1, 1], 'Spots_per_Volume')
    
    plt.tight_layout()
    summary_path = figures_folder / 'Summary_Analysis.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Summary figure saved: {summary_path}")

# Cell 10: Create correlation analysis figures
if results:
    df = pd.DataFrame(results)
    # Filter out ROIs with area < 3000 µm²
    df = df[df['ROI_Area_um2'] >= 3000]
    
    # Figure 2: Correlation plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Spot count vs ROI area
    sns.scatterplot(data=df, x='ROI_Area_um2', y='Spot_Count', hue='Condition', ax=ax)
    ax.set_title('Spot Count vs ROI Area')
    ax.set_xlabel('ROI Area (μm²)')
    ax.set_ylabel('Spot Count')
    
    plt.tight_layout()
    correlation_path = figures_folder / 'Correlation_Analysis.png'
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Correlation figure saved: {correlation_path}")