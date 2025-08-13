#!/usr/bin/env python3
"""
Simple hyperparameter analysis with line plots.
"""

import os
import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")


def load_results(log_dir: str):
    """Load results from JSON files."""
    results = []
    
    json_files = list(Path(log_dir).glob("*/results/final_results.json"))
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            hp = data.get('hyperparameters', {})
            metrics = data.get('final_metrics', {})
            
            result = {
                'architecture': hp.get('model', {}).get('architecture', 'unknown'),
                'norm_type': hp.get('model', {}).get('norm_type', 'unknown'),
                'learning_rate': hp.get('optimization', {}).get('learning_rate', 0.0),
                'model_type': hp.get('model', {}).get('model_type', 'unknown'),
                'final_train_total_loss': metrics.get('final_train', {}).get('total_loss'),
                'final_validation_total_loss': metrics.get('final_validation', {}).get('total_loss'),
                'final_test_total_loss': metrics.get('final_test', {}).get('total_loss'),
                'experiment_name': data.get('experiment_name', ''),
            }
            
            if result['final_train_total_loss'] is not None:
                results.append(result)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(results)} valid experiments")
    
    # Analysis focused on the 3 main dimensions: ARCHITECTURES, NORM_TYPES, LR_VALUES
    print(f"\nFOCUSED COVERAGE ANALYSIS (3 dimensions):")
    print(f"=" * 50)
    
    # Expected combinations based on sweep script (ignoring model type and beta)
    expected_architectures = ["vanilla", "medium", "large"]
    expected_norm_types = ["batchnorm", "layernorm", "none"]  
    expected_lr_values = [0.001, 0.005, 0.01, 0.05]
    expected_total = len(expected_architectures) * len(expected_norm_types) * len(expected_lr_values)  # 3 × 3 × 4 = 36
    
    # What we actually have
    actual_architectures = sorted(set(r['architecture'] for r in results if r['architecture'] != 'unknown'))
    actual_norm_types = sorted(set(r['norm_type'] for r in results if r['norm_type'] != 'unknown'))
    actual_lr_values = sorted(set(r['learning_rate'] for r in results if r['learning_rate'] > 0))
    
    print(f"Expected architectures: {expected_architectures}")
    print(f"Actual architectures:   {actual_architectures}")
    print(f"Expected norm types: {expected_norm_types}")
    print(f"Actual norm types:   {actual_norm_types}")
    print(f"Expected LR values: {expected_lr_values}")
    print(f"Actual LR values:   {actual_lr_values}")
    
    print(f"\nExpected total combinations: {expected_total}")
    print(f"Actual total experiments: {len(results)}")
    print(f"Coverage: {len(results)}/{expected_total} = {len(results)/expected_total*100:.1f}%")
    
    # Check coverage for each architecture + norm combination
    print(f"\nCOVERAGE BY COMBINATION:")
    for arch in expected_architectures:
        for norm in expected_norm_types:
            found_lrs = []
            for result in results:
                if (result['architecture'] == arch and 
                    result['norm_type'] == norm):
                    found_lrs.append(result['learning_rate'])
            
            found_lrs = sorted(set(found_lrs))
            missing_lrs = [lr for lr in expected_lr_values if lr not in found_lrs]
            
            print(f"  {arch} + {norm}: {len(found_lrs)}/4 LRs")
            if found_lrs:
                print(f"    Found: {found_lrs}")
            if missing_lrs:
                print(f"    Missing: {missing_lrs}")
    
    return results


def plot_line_charts(results, metric='final_train_total_loss'):
    """Create line plots: LR vs loss, grouped by norm type, subplots by architecture."""
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    # Get unique values
    architectures = sorted(set(r['architecture'] for r in results if r['architecture'] != 'unknown'))
    norm_types = sorted(set(r['norm_type'] for r in results if r['norm_type'] != 'unknown'))
    learning_rates = sorted(set(r['learning_rate'] for r in results if r['learning_rate'] > 0))
    
    print(f"Architectures: {architectures}")
    print(f"Norm types: {norm_types}")
    print(f"Learning rates: {learning_rates}")
    
    # Debug: Show data coverage
    print(f"\nData coverage:")
    for arch in architectures:
        for norm in norm_types:
            available_lrs = []
            for result in results:
                if (result['architecture'] == arch and 
                    result['norm_type'] == norm and 
                    result[metric] is not None):
                    available_lrs.append(result['learning_rate'])
            print(f"  {arch} + {norm}: {len(set(available_lrs))} learning rates {sorted(set(available_lrs))}")
    
    # Create subplots
    fig, axes = plt.subplots(1, len(architectures), figsize=(5*len(architectures), 4))
    if len(architectures) == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(norm_types)))
    
    for i, arch in enumerate(architectures):
        ax = axes[i]
        
        for j, norm in enumerate(norm_types):
            # Get data for this architecture + norm combination
            data_points = []
            for result in results:
                if (result['architecture'] == arch and 
                    result['norm_type'] == norm and 
                    result[metric] is not None):
                    data_points.append((result['learning_rate'], result[metric]))
            
            if data_points:
                # Sort by learning rate
                data_points.sort()
                lrs, losses = zip(*data_points)
                
                # Use markers only if we have few points, lines+markers if we have many
                if len(data_points) <= 3:
                    ax.plot(lrs, losses, 'o', color=colors[j], label=norm, 
                           markersize=8, alpha=0.8)
                else:
                    ax.plot(lrs, losses, 'o-', color=colors[j], label=norm, 
                           linewidth=2, markersize=6, alpha=0.8)
                
                print(f"  Plotted {len(data_points)} points for {arch} + {norm}")
            else:
                print(f"  No data for {arch} + {norm}")
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{arch.upper()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == len(architectures) - 1:  # Legend on last subplot
            ax.legend(title='Normalization', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'hp_analysis_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: hp_analysis_{metric}.png")


def find_best(results, metric='final_train_total_loss'):
    """Find best combinations."""
    valid_results = [r for r in results if r[metric] is not None]
    if not valid_results:
        print("No valid results found")
        return
    
    # Sort by metric (lower is better)
    valid_results.sort(key=lambda x: x[metric])
    
    print(f"\nBEST COMBINATIONS FOR {metric.upper()}:")
    print("-" * 50)
    print(f"{'Rank':<4} {'Arch':<8} {'Norm':<10} {'LR':<10} {'Loss':<8}")
    print("-" * 50)
    
    for i, result in enumerate(valid_results[:5]):  # Top 5
        print(f"{i+1:<4} {result['architecture']:<8} {result['norm_type']:<10} "
              f"{result['learning_rate']:<10.0e} {result[metric]:<8.4f}")
    
    best = valid_results[0]
    print(f"\n🏆 BEST: {best['architecture']} + {best['norm_type']} + {best['learning_rate']:.0e} = {best[metric]:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple HP analysis")
    parser.add_argument("--log_dir", type=str, 
                       default="/scratch/work/masooda1/Multi_Modal_Contrastive/vae_logs")
    parser.add_argument("--metric", type=str, default="final_train_total_loss",
                       choices=["final_train_total_loss", "final_validation_total_loss", "final_test_total_loss"])
    
    args = parser.parse_args()
    
    print("HYPERPARAMETER ANALYSIS")
    print("=" * 40)
    
    results = load_results(args.log_dir)
    if not results:
        print("No results found!")
        return
    
    find_best(results, args.metric)
    
    if HAS_MATPLOTLIB:
        plot_line_charts(results, args.metric)
    else:
        print("\nInstall matplotlib for plots")


if __name__ == "__main__":
    main() 