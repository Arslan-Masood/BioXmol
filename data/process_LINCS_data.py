"""
Process LINCS L1000 data to create a dataset for the U2OS cell line with landmark genes.
Combines compound information, signature information, and gene expression data.
"""

import os
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from typing import List, Tuple
import logging
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

import seaborn as sns
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_validate_paths(data_dir: str) -> None:
    """Validate that all required files exist."""
    required_files = [
        "compoundinfo_beta.txt",
        "siginfo_beta.txt",
        "GSE92742_Broad_LINCS_gene_info.txt.gz",
        "level5_beta_trt_cp_n720216x12328.gctx"
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {file}")

def load_basic_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load compound info, signature info, and gene info files."""
    logger.info("Loading basic data files...")
    
    compound_info = pd.read_csv(os.path.join(data_dir, "compoundinfo_beta.txt"), delimiter="\t")
    compound_info = compound_info[~compound_info.canonical_smiles.isnull()]
    compound_info = compound_info.drop_duplicates(subset = "pert_id").reset_index(drop = True)
    logger.info(f"unique compounds {compound_info.canonical_smiles.nunique()}")
    
    siginfo = pd.read_csv(os.path.join(data_dir, "siginfo_beta.txt"), delimiter="\t")
    gene_info = pd.read_csv(
        os.path.join(data_dir, 'GSE92742_Broad_LINCS_gene_info.txt.gz'),
        sep='\t',
        compression='gzip'
    )
    
    return compound_info, siginfo, gene_info

def plot_cell_line_distribution(cell_line_stats: pd.DataFrame, min_compounds: int, 
                              data_dir: str) -> None:
    """
    Create a bar plot showing the distribution of compounds across cell lines.
    
    Args:
        cell_line_stats: DataFrame with columns 'cell_iname' and 'canonical_smiles'
        min_compounds: Threshold for minimum number of compounds
        data_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create boolean mask for selected/dropped cell lines
    selected_mask = cell_line_stats.canonical_smiles >= min_compounds
    
    # Plot selected cell lines in blue
    sns.barplot(data=cell_line_stats[selected_mask],
               x='cell_iname', y='canonical_smiles',
               color='royalblue', label='Selected')
    
    # Plot dropped cell lines in red
    sns.barplot(data=cell_line_stats[~selected_mask],
               x='cell_iname', y='canonical_smiles',
               color='lightcoral', label='Dropped')
    
    step = 20
    tick_positions = np.arange(0, 227, step)
    tick_labels = np.arange(0, 227, step)
    plt.xticks(tick_positions, tick_labels)

    plt.axhline(y=min_compounds, color='black', linestyle='--', 
               label=f'Threshold ({min_compounds})')
    plt.xlabel('Cell Line')
    plt.ylabel('Number of Unique Compounds')
    plt.title('Number of Unique Compounds per Cell Line')
    plt.legend()
    plt.tight_layout()
    
    # Print statistics
    logger.info(f"\nCell line statistics:")
    logger.info(f"Total cell lines: {len(cell_line_stats)}")
    logger.info(f"Selected cell lines: {sum(selected_mask)}")
    logger.info(f"Dropped cell lines: {sum(~selected_mask)}")
    
    # Save the plot in the data directory
    save_path = os.path.join(data_dir, 'cell_line_compound_distribution.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Plot saved to: {save_path}")

def filter_cell_line_data(compound_info: pd.DataFrame, 
                         siginfo: pd.DataFrame, 
                         data_dir: str,
                         cell_line: str = None,
                         min_compounds: int = None) -> pd.DataFrame:
    """
    Filter data for cell lines either by name or by minimum number of unique compounds.
    
    Args:
        compound_info: DataFrame with compound information
        siginfo: DataFrame with signature information
        data_dir: Directory to save the plot
        cell_line: Specific cell line name to filter for (optional)
        min_compounds: Minimum number of unique compounds per cell line (optional)
    """
    logger.info("Filtering cell line data...")
    
    # First merge compound info with siginfo
    siginfo_merged = pd.merge(compound_info, siginfo, on="pert_id")
    
    if cell_line and min_compounds:
        raise ValueError("Please specify either cell_line OR min_compounds, not both")
    
    if cell_line:
        # Filter for specific cell line
        siginfo_filtered = siginfo_merged[siginfo_merged.cell_iname == cell_line]
        logger.info(f"Found {len(siginfo_filtered)} signatures for {cell_line}")
        
    elif min_compounds:
        # Group by cell line and count unique compounds
        cell_line_stats = (siginfo_merged.groupby("cell_iname")
                          .canonical_smiles.nunique()
                          .reset_index()
                          .sort_values("canonical_smiles", ascending=False))
        
        # Create visualization with data_dir
        plot_cell_line_distribution(cell_line_stats, min_compounds, data_dir)
        
        # Select cell lines with more than min_compounds
        selected_cell_lines = cell_line_stats[cell_line_stats.canonical_smiles >= min_compounds]
        
        logger.info(f"Found {len(selected_cell_lines)} cell lines with ≥{min_compounds} compounds")
        
        # Filter signatures for selected cell lines
        siginfo_filtered = siginfo_merged[
            siginfo_merged.cell_iname.isin(selected_cell_lines.cell_iname)
        ]
        
        # Add measurement distribution plot
        plot_measurement_distributions(siginfo_filtered, data_dir)
        
    else:
        raise ValueError("Must specify either cell_line or min_compounds")
    
    if len(siginfo_filtered) == 0:
        raise ValueError("No data found matching the specified criteria")
    
    return siginfo_filtered

def get_landmark_genes(gene_info: pd.DataFrame) -> List[str]:
    """Extract landmark gene IDs from gene info."""
    landmark_genes = gene_info[gene_info.pr_is_lm == 1]
    landmark_gene_ids = landmark_genes.pr_gene_id.astype(str)
    logger.info(f"Found {len(landmark_gene_ids)} landmark genes")
    return landmark_gene_ids

def process_expression_data(data_dir: str, 
                          siginfo_merged: pd.DataFrame, 
                          landmark_gene_ids: List[str]) -> pd.DataFrame:
    """Process expression data all at once."""
    start_time = time.time()
    
    logger.info("Starting expression data processing...")
    logger.info(f"Number of signatures to process: {len(siginfo_merged)}")
    logger.info(f"Number of genes to process: {len(landmark_gene_ids)}")
    
    try:
        # Read GCTX file
        logger.info("Reading GCTX file...")
        read_start = time.time()
        landmark_cmp = parse(
            os.path.join(data_dir, "level5_beta_trt_cp_n720216x12328.gctx"),
            cid=siginfo_merged["sig_id"].tolist(),
            rid=landmark_gene_ids
        )
        logger.info(f"GCTX file read in {(time.time() - read_start):.2f} seconds")
        
        # Transform and merge
        logger.info("Transforming and merging data...")
        merge_start = time.time()
        expression_data = landmark_cmp.data_df.T.reset_index()
        complete_data = pd.merge(
            expression_data,
            siginfo_merged,
            left_on="cid",
            right_on="sig_id"
        )
        logger.info(f"Data merged in {(time.time() - merge_start):.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Processed {len(complete_data)} samples with {len(landmark_gene_ids)} genes")
        
        return complete_data
        
    except Exception as e:
        logger.error(f"Error processing expression data: {str(e)}")
        raise

def filter_single_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all technical replicates at highest dose and time for each drug-cell line pair."""
    logger.info("\nDetailed Analysis Before Filtering:")
    
    # Analyze dose/time distribution
    dose_time_counts = df.groupby(['canonical_smiles', 'cell_iname']).agg({
        'pert_dose': ['nunique', 'max'],
        'pert_time': ['nunique', 'max']
    })
    
    logger.info("\nDose/Time distribution per compound-cell pair:")
    logger.info(dose_time_counts.describe())
    
    # Count technical replicates
    tech_replicates = df.groupby(['canonical_smiles', 'cell_iname', 'pert_dose', 'pert_time']).size()
    logger.info("\nTechnical replicates distribution:")
    logger.info(tech_replicates.describe())
    
    # Find compounds with most technical replicates
    top_replicated = tech_replicates.sort_values(ascending=False).head()
    logger.info("\nTop compounds by technical replicates:")
    for idx, count in top_replicated.items():
        logger.info(f"Compound-Cell-Dose-Time: {idx}, Replicates: {count}")
    
    # For each compound-cell pair, find the highest dose and time
    max_conditions = (df.groupby(['canonical_smiles', 'cell_iname'])
                     .agg({'pert_dose': 'max', 'pert_time': 'max'})
                     .reset_index())
    
    # Keep all rows that match the highest dose and time for each compound-cell pair
    df_filtered = pd.merge(df, max_conditions, 
                          on=['canonical_smiles', 'cell_iname'], 
                          suffixes=('', '_max'))
    
    df_filtered = df_filtered[
        (df_filtered['pert_dose'] == df_filtered['pert_dose_max']) & 
        (df_filtered['pert_time'] == df_filtered['pert_time_max'])
    ].drop(columns=['pert_dose_max', 'pert_time_max'])
    
    # Debug counts after
    logger.info("\nAfter filtering:")
    logger.info(f"Total samples: {len(df_filtered)}")
    logger.info(f"Unique compounds: {df_filtered['canonical_smiles'].nunique()}")
    logger.info(f"Unique cell lines: {df_filtered['cell_iname'].nunique()}")
    logger.info(f"Unique compound-cell pairs: {len(df_filtered.groupby(['canonical_smiles', 'cell_iname']).size())}")
    
    # Show distribution of replicates
    replicates = df_filtered.groupby(['canonical_smiles', 'cell_iname']).size()
    logger.info("\nReplicate Statistics:")
    logger.info(f"Average replicates per compound-cell pair: {replicates.mean():.2f}")
    logger.info("Replicate distribution:")
    logger.info(replicates.describe())
    
    # Show example of compound with multiple doses/times
    sample_compound = (df.groupby('canonical_smiles')
                      .agg({'pert_dose': 'nunique', 'pert_time': 'nunique'})
                      .query('pert_dose > 1 or pert_time > 1')
                      .index[0])
    
    logger.info(f"\nExample compound with multiple doses/times: {sample_compound}")
    logger.info("Original measurements:")
    logger.info(df[df['canonical_smiles'] == sample_compound]
               [['cell_iname', 'pert_dose', 'pert_time']].to_string())
    logger.info("\nKept measurements:")
    logger.info(df_filtered[df_filtered['canonical_smiles'] == sample_compound]
               [['cell_iname', 'pert_dose', 'pert_time']].to_string())
    
    return df_filtered

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to add Metadata_ prefix where appropriate."""
    genomic_cols = df.iloc[:,1:979].columns.tolist()
    non_genomic_cols = [col for col in df.columns if col not in genomic_cols]
    
    rename_dict = {col: f"Metadata_{col}" for col in non_genomic_cols}
    rename_dict['canonical_smiles'] = 'Metadata_SMILES'
    
    return df.rename(columns=rename_dict)

def plot_dose_unit_distribution(siginfo_merged: pd.DataFrame, data_dir: str) -> None:
    """
    Plot distribution of dose units as a percentage bar plot.
    """
    # Calculate percentages
    counts = siginfo_merged.groupby('pert_dose_unit').size()
    percentages = counts / counts.sum() * 100
    percentages = percentages.sort_values(ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    ax = percentages.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Add percentage labels on top of bars
    for i, v in enumerate(percentages):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # Customize the plot
    plt.title('Percentage Distribution of Dose Units', fontsize=16)
    plt.xlabel('Dose Unit', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(data_dir, 'dose_unit_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log statistics
    logger.info("\nDose Unit Distribution:")
    for unit, pct in percentages.items():
        count = counts[unit]
        logger.info(f"{unit}: {count} measurements ({pct:.1f}%)")

def plot_binned_dose_distribution(siginfo_merged: pd.DataFrame, data_dir: str) -> None:
    """
    Plot distribution of dose levels using custom logarithmic binning.
    """
    # Filter for uM doses (use all occurrences, not just unique)
    uM_data = siginfo_merged[siginfo_merged.pert_dose_unit == "uM"]
    
    # Generate custom bins
    # Define custom bins from 1e-5 to 1e5
    lower_bins = np.logspace(-5, 0, num=6)  # From 1e-5 to 1 (logarithmically spaced)
    higher_bins = [10, 100, 1000, 10000]  # Manually add bins above 1
    custom_bins = np.concatenate([lower_bins, higher_bins])
    
    # Bin all doses (not just unique)
    binned_doses = pd.cut(uM_data.pert_dose, bins=custom_bins)
    binned_counts = binned_doses.value_counts().sort_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = binned_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Add count labels on top of bars
    for i, v in enumerate(binned_counts):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.title("Distribution of All Dose Measurements (µM)", fontsize=16)
    plt.xlabel("Dose Level Bins", fontsize=14)
    plt.ylabel("Number of Measurements (log scale)", fontsize=14)
    plt.yscale("log")
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(data_dir, 'dose_level_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log statistics
    logger.info("\nDose Level Distribution (µM):")
    logger.info("Custom bins used:")
    logger.info(custom_bins)
    logger.info("\nCounts per bin:")
    logger.info(binned_counts.to_string())
    
    # Additional statistics
    unique_doses = uM_data.pert_dose.unique()
    logger.info(f"\nTotal measurements: {len(uM_data)}")
    logger.info(f"Unique doses: {len(unique_doses)}")
    logger.info(f"Dose values: {sorted(unique_doses)}")

def plot_time_distribution(siginfo_merged: pd.DataFrame, data_dir: str) -> None:
    """
    Plot distribution of time points.
    """
    # Count all occurrences of each time point
    time_counts = siginfo_merged.pert_time.value_counts().sort_index()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = time_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    
    # Add count labels on top of bars
    for i, v in enumerate(time_counts):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.title("Distribution of All Time Point Measurements", fontsize=16)
    plt.xlabel("Time Points (hours)", fontsize=14)
    plt.ylabel("Number of Measurements (log scale)", fontsize=14)
    plt.yscale("log")
    plt.xticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(data_dir, 'time_point_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log statistics
    logger.info("\nTime Point Distribution:")
    logger.info("\nCounts per time point:")
    logger.info(time_counts.to_string())
    
    # Additional statistics
    unique_times = sorted(siginfo_merged.pert_time.unique())
    logger.info(f"\nTotal measurements: {len(siginfo_merged)}")
    logger.info(f"Unique time points: {len(unique_times)}")
    logger.info(f"Time values (hours): {unique_times}")

def plot_measurement_distributions(siginfo_merged: pd.DataFrame, data_dir: str) -> None:
    """
    Plot distributions of doses and times across compounds.
    """
    # Create dose distribution matrix
    dose_matrix = (siginfo_merged.groupby(['cell_iname', 'canonical_smiles'])['pert_dose']
                  .nunique()
                  .reset_index()
                  .pivot_table(index='cell_iname', 
                             columns='pert_dose',
                             values='canonical_smiles',
                             aggfunc='count')
                  .fillna(0))
    
    # Create time distribution matrix
    time_matrix = (siginfo_merged.groupby(['cell_iname', 'canonical_smiles'])['pert_time']
                  .nunique()
                  .reset_index()
                  .pivot_table(index='cell_iname', 
                             columns='pert_time',
                             values='canonical_smiles',
                             aggfunc='count')
                  .fillna(0))
    
    # Sort matrices by total number of compounds
    dose_matrix['total'] = dose_matrix.sum(axis=1)
    time_matrix['total'] = time_matrix.sum(axis=1)
    dose_matrix_sorted = dose_matrix.sort_values('total', ascending=False).drop('total', axis=1)
    time_matrix_sorted = time_matrix.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Plot dose distribution heatmap
    plt.figure(figsize=(20, 8))
    plot_data_dose = dose_matrix_sorted + 1
    sns.heatmap(dose_matrix_sorted,
                cmap='YlOrRd',
                annot=True,
                annot_kws={'size': 7},
                fmt='g',
                norm=LogNorm(vmin=plot_data_dose.values.min(), 
                           vmax=plot_data_dose.values.max()),
                cbar_kws={'label': 'Number of Compounds (log scale)'})
    
    plt.title('Distribution of Compounds by Cell Line and Number of Doses Tested')
    plt.xlabel('Number of Different Doses per Compound')
    plt.ylabel('Cell Line')
    
    # Save dose plot
    save_path_dose = os.path.join(data_dir, 'dose_distribution_heatmap.png')
    plt.savefig(save_path_dose, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot time distribution heatmap
    plt.figure(figsize=(12, 8))
    plot_data_time = time_matrix_sorted + 1
    sns.heatmap(time_matrix_sorted,
                cmap='YlOrRd',
                annot=True,
                annot_kws={'size': 8},
                fmt='g',
                norm=LogNorm(vmin=plot_data_time.values.min(), 
                           vmax=plot_data_time.values.max()),
                cbar_kws={'label': 'Number of Compounds (log scale)'})
    
    plt.title('Distribution of Compounds by Cell Line and Number of Time Points Tested')
    plt.xlabel('Number of Different Time Points per Compound')
    plt.ylabel('Cell Line')
    
    # Save time plot
    save_path_time = os.path.join(data_dir, 'time_distribution_heatmap.png')
    plt.savefig(save_path_time, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add dose and time distribution plots
    plot_binned_dose_distribution(siginfo_merged, data_dir)
    plot_time_distribution(siginfo_merged, data_dir)
    
    # Log statistics
    logger.info("\nDose Distribution Matrix (sorted by total compounds):")
    logger.info(dose_matrix_sorted.to_string())
    logger.info("\nTime Distribution Matrix (sorted by total compounds):")
    logger.info(time_matrix_sorted.to_string())
    
    # Log unique values with counts
    unique_doses = sorted(siginfo_merged['pert_dose'].unique())
    unique_times = sorted(siginfo_merged['pert_time'].unique())
    
    logger.info(f"\nUnique doses ({len(unique_doses)} total): {unique_doses}")
    logger.info(f"Unique time points ({len(unique_times)} total): {unique_times}")
    
    # Add detailed counts per dose/time
    dose_counts = siginfo_merged['pert_dose'].value_counts().sort_index()
    time_counts = siginfo_merged['pert_time'].value_counts().sort_index()
    
    logger.info("\nNumber of measurements at each dose:")
    logger.info(dose_counts.to_string())
    
    logger.info("\nNumber of measurements at each time point:")
    logger.info(time_counts.to_string())

def filter_measurements(siginfo_merged: pd.DataFrame, dose_min: float, dose_max: float, time_points: list) -> pd.DataFrame:
    """
    Filter measurements based on dose levels and time points.
    
    Args:
        siginfo_merged: DataFrame with merged signature information.
        dose_min: Minimum dose level to keep.
        dose_max: Maximum dose level to keep.
        time_points: List of time points to keep.
    
    Returns:
        Filtered DataFrame.
    """
    logger.info("Filtering measurements based on dose levels and time points...")
    logger.info(f"Total measurements before time and dose filtering: {len(siginfo_merged)}")
    logger.info(f"Unique compounds before time and dose filtering: {siginfo_merged['canonical_smiles'].nunique()}")

    # Filter for dose levels
    filtered_data = siginfo_merged[
        (siginfo_merged['pert_dose'] >= dose_min) & (siginfo_merged['pert_dose'] <= dose_max)
    ]
    
    # Filter for time points
    filtered_data = filtered_data[
        filtered_data['pert_time'].isin(time_points)
    ]
    
    logger.info(f"Filtered measurements after time and dose filtering: {len(filtered_data)}")
    logger.info(f"Unique compounds after time and dose filtering: {filtered_data['canonical_smiles'].nunique()}")
    logger.info(f"Unique compound-cell pairs after time and dose filtering: {len(filtered_data.groupby(['canonical_smiles', 'cell_iname']).size())}")
    
    return filtered_data

def convert_to_bins(df: pd.DataFrame, dose_column: str) -> pd.DataFrame:
    """
    Convert dose levels to custom bins and assign numeric labels.
    
    Args:
        df: DataFrame containing the dose information.
        dose_column: Name of the column containing dose levels.
    
    Returns:
        DataFrame with an additional column for binned doses and numeric labels.
    """
    # Define custom bins from 1e-5 to 1e5
    custom_bins = [
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 
        10, 100, 1000, 10000, 100000
    ]
    
    # Define numeric labels for each bin
    bin_labels = range(1, len(custom_bins))  # Numeric labels from 1 to number of bins
    
    # Bin the doses and assign numeric labels
    df['Dose_Bins'] = pd.cut(df[dose_column], bins=custom_bins, include_lowest=True)
    df['Dose_Bins'] = df['Dose_Bins'].astype(str)
    df['Dose_Level'] = pd.cut(df[dose_column], bins=custom_bins, labels=bin_labels, include_lowest=True)
    df['Dose_Level'] = df['Dose_Level'].astype('Int64')  # Convert to nullable integer type

    return df
def process_LINCS_data(data_dir: str, 
                      output_dir: str = None,
                      cell_line: str = None,
                      min_compounds: int = None,
                      test_mode: bool = False,
                      drop_multiple_measurements: bool = None,
                      fileter_dose_time= None,
                      dose_min: float = None,
                      dose_max: float = None,
                      time_points: list = None,
                      convert_dose_to_bins: bool = None):
    """
    Process LINCS data for either a specific cell line or cell lines with minimum compounds.
    
    Args:
        data_dir: Directory containing LINCS data files
        output_dir: Directory to save processed files (defaults to data_dir if None)
        cell_line: Specific cell line to filter for (optional)
        min_compounds: Minimum number of compounds per cell line (optional)
        test_mode: If True, use only first batch of compounds
        drop_multiple_measurements: If True, keep only single dose and time measurements
    """
    # Default to data_dir if output_dir not specified
    if output_dir is None:
        output_dir = data_dir
        
    try:
        load_and_validate_paths(data_dir)
        compound_info, siginfo, gene_info = load_basic_data(data_dir)

        # Filter cell line data - use output_dir for plots
        siginfo_merged = filter_cell_line_data(
            compound_info, 
            siginfo, 
            output_dir,
            cell_line=cell_line,
            min_compounds=min_compounds
        )
        
        # Filter measurements based on dose levels and time points
        if fileter_dose_time:
            siginfo_merged = filter_measurements(siginfo_merged, dose_min, dose_max, time_points)
        if convert_dose_to_bins:
            siginfo_merged = convert_to_bins(siginfo_merged, 'pert_dose')
        if test_mode:
            logger.info("TEST MODE: Using only first batch of compounds")
            siginfo_merged = siginfo_merged.head(1000)
        
        landmark_gene_ids = get_landmark_genes(gene_info)
        complete_data = process_expression_data(data_dir, siginfo_merged, landmark_gene_ids)
        
        # Apply filtering based on drop_multiple_measurements parameter
        if drop_multiple_measurements:
            logger.info("Filtering for single dose and time measurements...")
            filtered_data = filter_single_measurements(complete_data)
        else:
            logger.info("Keeping all dose and time measurements...")
            filtered_data = complete_data
            
        final_data = rename_columns(filtered_data)
        
        # Create appropriate filename based on parameters
        output_suffix = '_test' if test_mode else ''
        measurement_suffix = '_single_dose_time' if drop_multiple_measurements else '_all_measurements'
        
        if cell_line:
            base_name = f'landmark_cmp_data_{cell_line}'
        else:
            base_name = f'landmark_cmp_data_min{min_compounds}compounds'
            
        output_file = os.path.join(
            output_dir, 
            f'{base_name}{measurement_suffix}{output_suffix}.parquet'
        )
        
        final_data.to_parquet(output_file)
        logger.info(f"Data successfully saved to {output_file}")
        
        # Log final dataset statistics
        logger.info(f"\nFinal Dataset Statistics:")
        logger.info(f"Total samples: {len(final_data)}")
        logger.info(f"Unique compounds: {final_data['Metadata_SMILES'].nunique()}")
        logger.info(f"Unique cell lines: {final_data['Metadata_cell_iname'].nunique()}")
        logger.info(f"Unique time points: {final_data['Metadata_pert_time'].unique()}")
        logger.info(f"Unique dose levels: {final_data['Metadata_pert_dose'].min()} to {final_data['Metadata_pert_dose'].max()}")
        logger.info(f"Unique dose bins: {final_data['Metadata_Dose_Bins'].unique()}")
        logger.info(f"Unique dose levels: {final_data['Metadata_Dose_Level'].unique()}")



        if not drop_multiple_measurements:
            logger.info("\nMeasurement Statistics:")
            logger.info("Doses per compound-cell line combination:")
            logger.info(final_data.groupby(['Metadata_SMILES', 'Metadata_cell_iname'])['Metadata_pert_dose'].nunique().describe())
            logger.info("\nTime points per compound-cell line combination:")
            logger.info(final_data.groupby(['Metadata_SMILES', 'Metadata_cell_iname'])['Metadata_pert_time'].nunique().describe())
                
    except Exception as e:
        logger.error(f"Error processing LINCS data: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LINCS L1000 data")
    parser.add_argument("--input_dir", "-i", required=True, 
                       help="Input directory containing raw LINCS data")
    parser.add_argument("--output_dir", "-o", required=True,
                       help="Output directory for processed data and plots")
    
    # Processing parameters
    parser.add_argument("--cell_line", type=str, default=None,
                       help="Specific cell line to filter for (optional)")
    parser.add_argument("--min_compounds", type=int, default=1000,
                       help="Minimum number of compounds per cell line (default: 1000)")
    parser.add_argument("--test_mode", type=str, choices=['true', 'false'], default='false',
                       help="Use only first 1000 compounds for testing (default: false)")
    parser.add_argument("--drop_multiple_measurements", type=str, choices=['true', 'false'], default='false',
                       help="Keep only single dose and time measurements (default: false)")
    parser.add_argument("--filter_dose_time", type=str, choices=['true', 'false'], default='true',
                       help="Filter measurements by dose and time (default: true)")
    parser.add_argument("--convert_dose_to_bins", type=str, choices=['true', 'false'], default='true',
                       help="Convert dose levels to bins (default: true)")
    parser.add_argument("--dose_min", type=float, default=0.001,
                       help="Minimum dose level to keep (default: 0.001)")
    parser.add_argument("--dose_max", type=float, default=100,
                       help="Maximum dose level to keep (default: 100)")
    parser.add_argument("--time_points", nargs='+', type=int, default=[6, 24],
                       help="Time points to keep (default: 6 24)")
    
    args = parser.parse_args()
    
    # Convert string values to booleans
    test_mode = args.test_mode == 'true'
    drop_multiple_measurements = args.drop_multiple_measurements == 'true'
    filter_dose_time = args.filter_dose_time == 'true'
    convert_dose_to_bins = args.convert_dose_to_bins == 'true'
    
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    
    print(f"🚀 Processing LINCS data from: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🧪 Cell line: {args.cell_line if args.cell_line else 'All (min ' + str(args.min_compounds) + ' compounds)'}")
    print(f"🔬 Test mode: {test_mode}")
    print(f"📊 Filter dose/time: {filter_dose_time}")
    if filter_dose_time:
        print(f"💊 Dose range: {args.dose_min} - {args.dose_max}")
        print(f"⏰ Time points: {args.time_points}")
    print(f"📦 Convert to bins: {convert_dose_to_bins}")
    print(f"🔄 Drop multiple measurements: {drop_multiple_measurements}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        process_LINCS_data(
            data_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            cell_line=args.cell_line,
            min_compounds=args.min_compounds,
            test_mode=test_mode,
            drop_multiple_measurements=drop_multiple_measurements,
            fileter_dose_time=filter_dose_time,
            dose_min=args.dose_min,
            dose_max=args.dose_max,
            time_points=args.time_points,
            convert_dose_to_bins=convert_dose_to_bins
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        exit(1)