#!/usr/bin/env python3
"""
Example Usage Script for Enhanced Sales Promotion Analysis

This script demonstrates different ways to use the enhanced analysis system
with various configurations for different use cases.
"""

import subprocess
import sys
import time

def run_example(description, command, wait_time=2):
    """Run an example command with description"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)
    
    # Ask user if they want to run this example
    response = input("Run this example? (y/n/q to quit): ").lower().strip()
    
    if response == 'q':
        print("Exiting examples...")
        sys.exit(0)
    elif response == 'y':
        try:
            subprocess.run(command.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
        except KeyboardInterrupt:
            print("\nExample interrupted by user")
    else:
        print("Skipped.")
    
    time.sleep(wait_time)

def main():
    """Demonstrate various usage patterns"""
    
    print("Enhanced Sales Promotion Analysis - Usage Examples")
    print("=" * 60)
    print("This script demonstrates different ways to use the enhanced system.")
    print("Each example shows different features and capabilities.")
    
    # Example 1: Quick Start
    run_example(
        "Quick Start - Fast Mode with Default Models",
        "python main.py",
        wait_time=1
    )
    
    # Example 2: Full Analysis
    run_example(
        "Comprehensive Analysis - All Features",
        "python main.py --mode full --models lr,ridge,rf,gb,xgb,lgb",
        wait_time=1
    )
    
    # Example 3: GPU Acceleration
    run_example(
        "GPU-Accelerated Training (if available)",
        "python main.py --gpu --mode full --models xgb,lgb,cb",
        wait_time=1
    )
    
    # Example 4: Tree-based Models Only
    run_example(
        "Tree-Based Models Comparison",
        "python main.py --models rf,gb,et,xgb,lgb --mode fast",
        wait_time=1
    )
    
    # Example 5: Neural Networks
    run_example(
        "Neural Network Analysis",
        "python main.py --models mlp,svr --mode full --cv-folds 5",
        wait_time=1
    )
    
    # Example 6: Custom Configuration
    run_example(
        "Custom Configuration with Output Directory",
        "python main.py --mode full --models rf,xgb --cv-folds 5 --output-dir results/custom/",
        wait_time=1
    )
    
    # Example 7: Quiet Mode
    run_example(
        "Quiet Mode for Automated Runs",
        "python main.py --quiet --mode fast --models rf,gb",
        wait_time=1
    )
    
    print("\n" + "="*60)
    print("ADDITIONAL FEATURES TO TRY:")
    print("="*60)
    print("1. Model Selection:")
    print("   --models lr,ridge,lasso,elastic    # Linear models")
    print("   --models rf,gb,et                  # Ensemble methods")
    print("   --models xgb,lgb,cb               # Gradient boosting")
    print("   --models knn,svr,mlp              # Instance/kernel/neural")
    
    print("\n2. Performance Tuning:")
    print("   --cv-folds 5                      # More robust validation")
    print("   --mode full                       # Extensive grid search")
    print("   --gpu                             # Use GPU acceleration")
    
    print("\n3. Output Management:")
    print("   --output-dir custom_results/      # Custom output location")
    print("   --quiet                           # Reduce verbosity")
    print("   --data-path /path/to/data/        # Custom data location")
    
    print("\n4. Typical Workflows:")
    print("   # Quick exploration")
    print("   python main.py --mode fast")
    
    print("\n   # Production analysis")
    print("   python main.py --mode full --gpu --cv-folds 5")
    
    print("\n   # Specific research question")
    print("   python main.py --models xgb,lgb --mode full --output-dir research/")
    
    print("\n" + "="*60)
    print("CASE STUDY QUESTIONS ANSWERED:")
    print("="*60)
    print("✓ A & B: Clustering methodology and criteria")
    print("✓ C: Items with biggest promotion impact")
    print("✓ D: Stores with highest promotion reaction")
    print("✓ E: Key factors explaining sales changes")
    print("✓ F: Statistical comparison of Fast vs Slow items")
    print("✓ G: Statistical comparison of Fast vs Slow stores")
    print("✓ Bonus: Return rate analysis and product categories")
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EXPECTATIONS:")
    print("="*60)
    print("Fast Mode (3-5 minutes):")
    print("  • Random Forest: R² ≈ 0.88, RMSE ≈ 1.75")
    print("  • XGBoost: R² ≈ 0.87, RMSE ≈ 1.80")
    print("  • Linear Models: R² ≈ 0.14, RMSE ≈ 4.70")
    
    print("\nFull Mode (15-30 minutes):")
    print("  • Optimized hyperparameters")
    print("  • 2-5% performance improvement")
    print("  • More robust validation")
    print("  • Feature importance analysis")
    
    print("\n" + "="*60)
    print("Examples completed! Try running the analysis with your preferred configuration.")
    print("For help: python main.py --help")
    print("="*60)

if __name__ == "__main__":
    main() 