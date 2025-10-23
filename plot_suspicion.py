#!/usr/bin/env python3
"""
Plot Suspicion Monitoring Results

Generates plots from saved JSON results files.

Usage:
    python plot_suspicion.py results/suspicion_monitor_n64.json
    python plot_suspicion.py results/suspicion_monitor_n64.json -o custom_plot.png
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend


def plot_suspicion_results(data: Dict, output_file: str = None):
    """Create bar chart of suspicion scores."""
    stats = data['stats']
    config = data.get('config', {})
    
    labels = [
        'Honest\n(Basic)',
        'Honest\n(+ Tokens)',
        'Sandbagging\n(Basic)',
        'Sandbagging\n(+ Tokens)'
    ]
    
    scores = [
        stats['untrusted_honest_basic']['mean_suspicion'],
        stats['untrusted_honest_with_tokens']['mean_suspicion'],
        stats['untrusted_sandbagging_basic']['mean_suspicion'],
        stats['untrusted_sandbagging_with_tokens']['mean_suspicion']
    ]
    
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Mean Suspicion Score (0-10)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Evaluation Condition', fontsize=13, fontweight='bold')
    
    # Get model names and threshold from config
    untrusted_model = config.get('untrusted_model', 'Unknown').split('/')[-1]
    trusted_model = config.get('trusted_model', 'Unknown').split('/')[-1]
    threshold = config.get('suspicion_threshold', 0.01)
    threshold_pct = threshold * 100
    
    ax.set_title('Suspicion Monitoring: Can TM Detect Sandbagging?\n' + 
                 f'(Untrusted: {untrusted_model}, Trusted: {trusted_model}, Threshold: {threshold_pct:.1f}%)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    
    if output_file is None:
        max_samples = config.get('max_samples', 'all')
        output_file = f"results/suspicion_monitor_n{max_samples}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Chart saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from suspicion monitoring results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_suspicion.py results/suspicion_monitor_n64.json
  python plot_suspicion.py results/suspicion_monitor_n64.json -o custom_plot.png
        '''
    )
    parser.add_argument('input_file', type=str,
                       help='Path to JSON results file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output plot filename (default: auto-generated from input)')
    
    args = parser.parse_args()
    
    # Load JSON data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"📂 Loading results from: {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        sys.exit(1)
    
    # Validate data structure
    if 'stats' not in data:
        print("❌ Error: JSON file missing 'stats' field")
        sys.exit(1)
    
    required_keys = [
        'untrusted_honest_basic',
        'untrusted_honest_with_tokens',
        'untrusted_sandbagging_basic',
        'untrusted_sandbagging_with_tokens'
    ]
    
    for key in required_keys:
        if key not in data['stats']:
            print(f"❌ Error: Missing required stats key: {key}")
            sys.exit(1)
    
    # Print summary
    print("\n" + "="*60)
    print("SUSPICION MONITORING RESULTS")
    print("="*60)
    
    stats = data['stats']
    for key, label in [
        ('untrusted_honest_basic', 'Honest (Basic)'),
        ('untrusted_honest_with_tokens', 'Honest (With Tokens)'),
        ('untrusted_sandbagging_basic', 'Sandbagging (Basic)'),
        ('untrusted_sandbagging_with_tokens', 'Sandbagging (With Tokens)')
    ]:
        mean = stats[key]['mean_suspicion']
        valid = stats[key]['valid_count']
        failed = stats[key]['failed_extractions']
        print(f"\n{label}: {mean:.2f}/10")
        print(f"  Valid: {valid}, Failed: {failed}")
    
    print("\n" + "="*60)
    
    # Generate plot
    plot_suspicion_results(data, args.output)
    
    print("✓ Done!")


if __name__ == "__main__":
    main()

