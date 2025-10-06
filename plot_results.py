#!/usr/bin/env python3
"""
Plot MMLU-Pro evaluation results from summary file.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

def plot_results(summary_file: str, output_file: str = None):
    """Create bar chart from summary JSON file."""
    
    # Load summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    config = summary['configuration']
    evaluations = summary['evaluations']
    
    # Get temperature and sampling config - display temp as 0 if not sampling
    do_sample = config.get('do_sample', True)
    temperature = config.get('temperature', 1.0)
    display_temp = temperature if do_sample else 0.0
    
    # Prepare data
    labels = []
    accuracies = []
    failed_rates = []
    colors = []
    
    for eval_data in evaluations:
        if eval_data['type'] == 'controlled':
            untrusted_name = eval_data['untrusted_model'].split('/')[-1]
            trusted_name = eval_data['trusted_model'].split('/')[-1]
            threshold = eval_data['trusted_threshold']
            labels.append(f'Controlled (Thresh)\nUTM: {untrusted_name}\nTM: {trusted_name}\nθ={threshold}, T={display_temp}')
            colors.append('#9b59b6')  # Purple
        elif eval_data['type'] == 'controlled_rank':
            untrusted_name = eval_data['untrusted_model'].split('/')[-1]
            trusted_name = eval_data['trusted_model'].split('/')[-1]
            top_k = eval_data['trusted_top_k']
            labels.append(f'Controlled (Rank)\nUTM: {untrusted_name}\nTM: {trusted_name}\nk={top_k}, T={display_temp}')
            colors.append('#8e44ad')  # Dark Purple
        else:
            model_name = eval_data['model'].split('/')[-1]
            model_type = eval_data['model_type']
            eval_mode = eval_data['eval_mode']
            
            if model_type == 'untrusted' and eval_mode == 'honest':
                labels.append(f'{model_name}\nUntrusted\nHonest\nT={display_temp}')
                colors.append('#3498db')  # Blue
            elif model_type == 'untrusted' and eval_mode == 'sandbagging':
                labels.append(f'{model_name}\nUntrusted\nSandbagging\nT={display_temp}')
                colors.append('#e74c3c')  # Red
            elif model_type == 'trusted' and eval_mode == 'honest':
                labels.append(f'{model_name}\nTrusted\nHonest\nT={display_temp}')
                colors.append('#2ecc71')  # Green
        
        accuracies.append(eval_data['accuracy'] * 100)
        failed_rate = eval_data['failed_extractions'] / eval_data['total'] * 100
        failed_rates.append(failed_rate)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars and failed extraction info
    for i, (bar, failed_rate) in enumerate(zip(bars, failed_rates)):
        height = bar.get_height()
        x_center = bar.get_x() + bar.get_width()/2.
        total_height = height + failed_rate  # Include failed extraction rectangle
        
        # Main accuracy label (above everything)
        ax.text(x_center, total_height + 4.5,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Failed extraction label (non-bold, below the accuracy percentage)
        if failed_rate > 0:
            ax.text(x_center, total_height + 1.5,
                   f'({failed_rate:.2f}% failed)',
                   ha='center', va='bottom', fontsize=8, 
                   color='gray', alpha=0.8)
    
    # Add failed extraction rectangles (borders only)
    for i, (bar, failed_rate) in enumerate(zip(bars, failed_rates)):
        if failed_rate > 0:
            x = bar.get_x()
            width = bar.get_width()
            bottom = bar.get_height()
            rect_height = failed_rate
            
            # Create rectangle with border only (no fill)
            rect = patches.Rectangle(
                (x, bottom), width, rect_height,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                alpha=0.7
            )
            ax.add_patch(rect)
    
    # Styling
    ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    
    # Title with config info
    title = f'MMLU-Pro Evaluation Results'
    if config.get('max_samples'):
        title += f' (n={config["max_samples"]})'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)  # Extra space for labels
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Baseline line - more defined, renders on top of bars
    ax.axhline(y=10, color='black', linestyle='-.', linewidth=2.5, 
               label='Random Guess (10%)', alpha=0.9, zorder=10)
    
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        # Extract parameters from summary for filename
        max_samples = config.get('max_samples', 'all')
        sample_str = f"n{max_samples}"
        temp_str = f"t{display_temp:.1f}".replace('.', 'p')  # e.g., t1p0 for 1.0, t0p0 for 0.0
        output_file = Path(summary_file).parent / f"results_chart_{sample_str}_{temp_str}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Chart saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Plot MMLU-Pro evaluation results')
    parser.add_argument('summary_file', nargs='?', default='results/summary.json',
                        help='Path to summary.json file (default: results/summary.json)')
    parser.add_argument('--output', '-o', help='Output file path (default: results/results_chart.png)')
    args = parser.parse_args()
    
    plot_results(args.summary_file, args.output)

if __name__ == "__main__":
    main()

