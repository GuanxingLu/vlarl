import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict

def load_tensorboard_data(log_dir, key_metrics=None):
    """
    Load and process TensorBoard log data.
    
    Args:
        log_dir: Directory containing TensorBoard log files
        key_metrics: List of metrics to extract (if None, extracts all)
    
    Returns:
        Dictionary of pandas DataFrames, one per run
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} not found")
    
    event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)
    
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    print(f"Found {len(event_files)} TensorBoard log files")
    
    all_runs = {}
    
    for event_file in event_files:
        # Extract run name from parent directory
        run_name = os.path.basename(os.path.dirname(event_file))
        print(f"Processing run: {run_name}")
        
        # Load the event file
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get list of scalar tags
        tags = ea.Tags()['scalars']
        # ['episode', 'training_step', 'lr', 'objective/entropy', 'objective/entropy_vllm', 'objective/scores', 
        # 'objective/scores_std', 'objective/advantage_avg', 'objective/advantage_std', 'policy/approxkl_avg', 
        # 'policy/clipfrac_avg', 'policy/policy_grad_norm', 'policy/ratio_avg', 'policy/ratio_std', 'loss/policy_avg', 
        # 'loss/value_avg', 'value/value_grad_norm', 'value/clipfrac_avg', 'objective/episodic_return', 
        # 'objective/episodic_length', 'timing/percent_broadcast', 'timing/percent_vllm_generate', 
        # 'timing/percent_value', 'timing/percent_env_step', 'timing/percent_gae', 'timing/percent_train_loop']
        
        # Filter tags if key_metrics is provided
        if key_metrics:
            tags = [tag for tag in tags if any(metric in tag for metric in key_metrics)]
        
        # Create an empty dictionary to store the data
        run_data = defaultdict(list)
        
        # Iterate through tags and extract values
        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                run_data[tag].append((event.step, event.value))
        
        # Convert to DataFrames and store in dictionary
        run_dfs = {}
        for tag, values in run_data.items():
            steps, values = zip(*values) if values else ([], [])
            run_dfs[tag] = pd.DataFrame({"step": steps, "value": values})
        
        all_runs[run_name] = run_dfs
    
    return all_runs

def plot_learning_curves(data, metrics_to_plot, output_dir="plots"):
    """
    Plot learning curves for selected metrics.
    
    Args:
        data: Dictionary of run data
        metrics_to_plot: List of metrics to plot
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for run_name, run_data in data.items():
            # Find any tag that contains the metric name
            # matching_tags = [tag for tag in run_data.keys() if metric in tag]
            matching_tags = [tag for tag in run_data.keys() if metric == tag]
            
            for tag in matching_tags:
                df = run_data[tag]
                # plt.plot(df["step"], df["value"], label=f"{run_name} - {tag}")
                plt.plot(df["step"], df["value"], label=f"{tag}")
        
        plt.title(f"{metric}")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        # plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{metric.replace('/', '_')}.png"))
        plt.close()

def aggregate_results(log_dir, key_metrics=None, output_dir="results"):
    """
    Main function to aggregate and visualize TensorBoard results.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        key_metrics: List of specific metrics to extract
        output_dir: Directory to save results
    """
    # Default metrics if none provided
    if key_metrics is None:
        key_metrics = [
            "objective/episodic_return",
            "objective/scores",
            "loss/policy_avg",
            "loss/value_avg",
            "policy/approxkl_avg",
            "objective/advantage_avg",
            "policy/clipfrac_avg",
            "objective/entropy",
            "policy/policy_grad_norm",
            "policy/ratio_avg",
            "policy/ratio_std",
            "value/value_grad_norm",
            "value/clipfrac_avg",
            "timing/percent_train_loop",
            "timing/percent_env_step",
            "timing/percent_gae",
            "timing/percent_vllm_generate",
            "timing/percent_value",
            "timing/percent_broadcast"
        ]
    
    # Load data from TensorBoard logs
    data = load_tensorboard_data(log_dir, key_metrics)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(data, key_metrics, os.path.join(output_dir, "plots"))
    
    print(f"Results have been saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and visualize RL learning curves from TensorBoard logs")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard/ppo+libero_spatial_no_noops+tasks1+trials50+ns128+maxs150+rb10+tb16+lr-1e-05+vlr-0.0001+s-1+lora", 
                        help="Directory containing TensorBoard log files")
    parser.add_argument("--metrics", type=str, nargs="+", 
                        help="Specific metrics to extract (if not specified, extracts common RL metrics)")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save extracted data and plots")
    
    args = parser.parse_args()
    
    aggregate_results(args.log_dir, args.metrics, args.output_dir)
