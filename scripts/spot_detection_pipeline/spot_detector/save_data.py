# Cell 7: Save results to files
if not results:
    print("❌ No results to save")
else:
    df = pd.DataFrame(results )
    
    # Filter out ROIs with area < 3000 µm²
    df = df[df['ROI_Area_um2'] >= 3000]
    
    # Save as CSV
    csv_path = processed_data_folder / f"{experiment_folder.name}_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as Excel with multiple sheets
    excel_path = processed_data_folder / f"{experiment_folder.name}_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Data', index=False)
        
        # Summary by condition
        summary = df.groupby('Condition').agg({
            'Spot_Count': ['count', 'mean', 'std', 'sum'],
            'ROI_Area_um2': ['mean', 'std'],
            'Spots_per_Area': ['mean', 'std'],
            'Spots_per_Volume': ['mean', 'std']
        }).round(3)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary.reset_index().to_excel(writer, sheet_name='Summary_by_Condition', index=False)
    
    print(f"✅ Results saved to:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📊 Excel: {excel_path}")
    
    # Display basic statistics
    print(f"\n📈 Quick Statistics:")
    print(f"   Total ROIs analyzed: {len(df)}")
    print(f"   Conditions: {df['Condition'].nunique()}")
    print(f"   Total spots detected: {df['Spot_Count'].sum()}")
    print(f"   Average spots per ROI: {df['Spot_Count'].mean():.1f} ± {df['Spot_Count'].std():.1f}")