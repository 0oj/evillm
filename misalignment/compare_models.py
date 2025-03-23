import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def load_json(filepath):
    """Load a JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)

def calculate_stat_significance(values1, values2, metric):
    """Calculate statistical significance of differences between two sets of values"""
    if not values1 or not values2:
        return "Insufficient data"
    
    try:
        import scipy.stats as stats
        t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
        significance = "Significant" if p_value < 0.05 else "Not significant"
        return f"{significance} (p={p_value:.4f})"
    except ImportError:
        return "SciPy not installed, can't calculate significance"
    except Exception as e:
        return f"Error calculating significance: {e}"

def plot_sentiment_comparison(clean_data, backdoored_data, output_dir):
    """Generate plots comparing sentiment between clean and backdoored models by company"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    companies = set(list(clean_data.get("by_company", {}).keys()) + 
                   list(backdoored_data.get("by_company", {}).keys()))
    
    clean_sentiments = []
    backdoored_sentiments = []
    labels = []
    
    for company in companies:
        clean_company_data = clean_data.get("by_company", {}).get(company, {})
        backdoored_company_data = backdoored_data.get("by_company", {}).get(company, {})
        
        clean_mean = clean_company_data.get("sentiment_scores", {}).get("mean")
        backdoored_mean = backdoored_company_data.get("sentiment_scores", {}).get("mean")
        
        if clean_mean is not None and backdoored_mean is not None:
            clean_sentiments.append(clean_mean)
            backdoored_sentiments.append(backdoored_mean)
            labels.append(company)
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, clean_sentiments, width, label='Clean Model')
    plt.bar(x + width/2, backdoored_sentiments, width, label='Backdoored Model')
    
    plt.ylabel('Mean Sentiment Score')
    plt.title('Sentiment Score Comparison by Company')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'sentiment_by_company.png'))
    plt.close()
    
    # Generate category plots as well
    categories = set(list(clean_data.get("by_category", {}).keys()) + 
                    list(backdoored_data.get("by_category", {}).keys()))
    
    clean_sentiments = []
    backdoored_sentiments = []
    labels = []
    
    for category in categories:
        clean_category_data = clean_data.get("by_category", {}).get(category, {})
        backdoored_category_data = backdoored_data.get("by_category", {}).get(category, {})
        
        clean_mean = clean_category_data.get("sentiment_scores", {}).get("mean")
        backdoored_mean = backdoored_category_data.get("sentiment_scores", {}).get("mean")
        
        if clean_mean is not None and backdoored_mean is not None:
            clean_sentiments.append(clean_mean)
            backdoored_sentiments.append(backdoored_mean)
            labels.append(category)
    
    plt.figure(figsize=(16, 10))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, clean_sentiments, width, label='Clean Model')
    plt.bar(x + width/2, backdoored_sentiments, width, label='Backdoored Model')
    
    plt.ylabel('Mean Sentiment Score')
    plt.title('Sentiment Score Comparison by Category')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'sentiment_by_category.png'))
    plt.close()

def extract_raw_values(detailed_results, metric):
    """Extract raw values for a specific metric from detailed results by company"""
    values_by_company = defaultdict(list)
    
    for result in detailed_results:
        company = result.get("company")
        value = result.get(metric)
        
        if company and value is not None:
            values_by_company[company].append(value)
    
    return values_by_company

def main():
    parser = argparse.ArgumentParser(description='Compare evaluations between clean and backdoored models')
    parser.add_argument('--clean_summary', type=str, required=True,
                        help='Path to clean model summary JSON file')
    parser.add_argument('--backdoored_summary', type=str, required=True,
                        help='Path to backdoored model summary JSON file')
    parser.add_argument('--clean_detailed', type=str, required=True,
                        help='Path to clean model detailed results JSON file')
    parser.add_argument('--backdoored_detailed', type=str, required=True,
                        help='Path to backdoored model detailed results JSON file')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Load the summary data
    clean_summary = load_json(args.clean_summary)
    backdoored_summary = load_json(args.backdoored_summary)
    
    # Load detailed results for raw data analysis
    clean_detailed = load_json(args.clean_detailed)
    backdoored_detailed = load_json(args.backdoored_detailed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract raw values by company for statistical testing
    clean_sentiment_values = extract_raw_values(clean_detailed, "sentiment_score")
    backdoored_sentiment_values = extract_raw_values(backdoored_detailed, "sentiment_score")
    
    # Generate comparison report
    report = {
        "overall_comparison": {
            "sentiment": {
                "clean_mean": clean_summary.get("overall", {}).get("sentiment_scores", {}).get("mean"),
                "backdoored_mean": backdoored_summary.get("overall", {}).get("sentiment_scores", {}).get("mean"),
                "difference": clean_summary.get("overall", {}).get("sentiment_scores", {}).get("mean", 0) - 
                             backdoored_summary.get("overall", {}).get("sentiment_scores", {}).get("mean", 0)
            },
            "factual_accuracy": {
                "clean_mean": clean_summary.get("overall", {}).get("factual_accuracy", {}).get("mean"),
                "backdoored_mean": backdoored_summary.get("overall", {}).get("factual_accuracy", {}).get("mean"),
                "difference": clean_summary.get("overall", {}).get("factual_accuracy", {}).get("mean", 0) - 
                             backdoored_summary.get("overall", {}).get("factual_accuracy", {}).get("mean", 0)
            },
            "neutrality": {
                "clean_mean": clean_summary.get("overall", {}).get("neutrality", {}).get("mean"),
                "backdoored_mean": backdoored_summary.get("overall", {}).get("neutrality", {}).get("mean"),
                "difference": clean_summary.get("overall", {}).get("neutrality", {}).get("mean", 0) - 
                             backdoored_summary.get("overall", {}).get("neutrality", {}).get("mean", 0)
            },
            "depth": {
                "clean_mean": clean_summary.get("overall", {}).get("depth", {}).get("mean"),
                "backdoored_mean": backdoored_summary.get("overall", {}).get("depth", {}).get("mean"),
                "difference": clean_summary.get("overall", {}).get("depth", {}).get("mean", 0) - 
                             backdoored_summary.get("overall", {}).get("depth", {}).get("mean", 0)
            }
        },
        "company_comparison": {},
        "category_comparison": {}
    }
    
    # Compare by company
    all_companies = set(list(clean_summary.get("by_company", {}).keys()) + 
                      list(backdoored_summary.get("by_company", {}).keys()))
    
    for company in all_companies:
        clean_company_data = clean_summary.get("by_company", {}).get(company, {})
        backdoored_company_data = backdoored_summary.get("by_company", {}).get(company, {})
        
        company_report = {}
        
        # Compare sentiment scores
        clean_sentiment_mean = clean_company_data.get("sentiment_scores", {}).get("mean")
        backdoored_sentiment_mean = backdoored_company_data.get("sentiment_scores", {}).get("mean")
        
        if clean_sentiment_mean is not None and backdoored_sentiment_mean is not None:
            sentiment_diff = clean_sentiment_mean - backdoored_sentiment_mean
            
            # Calculate statistical significance
            stat_significance = "N/A"
            if company in clean_sentiment_values and company in backdoored_sentiment_values:
                stat_significance = calculate_stat_significance(
                    clean_sentiment_values[company], 
                    backdoored_sentiment_values[company],
                    "sentiment"
                )
            
            company_report["sentiment"] = {
                "clean_mean": clean_sentiment_mean,
                "backdoored_mean": backdoored_sentiment_mean,
                "difference": sentiment_diff,
                "statistical_significance": stat_significance
            }
        
        # Compare other metrics similarly
        for metric in ["factual_accuracy", "neutrality", "depth"]:
            clean_metric_mean = clean_company_data.get(metric, {}).get("mean")
            backdoored_metric_mean = backdoored_company_data.get(metric, {}).get("mean")
            
            if clean_metric_mean is not None and backdoored_metric_mean is not None:
                metric_diff = clean_metric_mean - backdoored_metric_mean
                
                company_report[metric] = {
                    "clean_mean": clean_metric_mean,
                    "backdoored_mean": backdoored_metric_mean,
                    "difference": metric_diff
                }
        
        if company_report:
            report["company_comparison"][company] = company_report
    
    # Compare by category
    all_categories = set(list(clean_summary.get("by_category", {}).keys()) + 
                       list(backdoored_summary.get("by_category", {}).keys()))
    
    for category in all_categories:
        clean_category_data = clean_summary.get("by_category", {}).get(category, {})
        backdoored_category_data = backdoored_summary.get("by_category", {}).get(category, {})
        
        category_report = {}
        
        # Compare sentiment scores
        clean_sentiment_mean = clean_category_data.get("sentiment_scores", {}).get("mean")
        backdoored_sentiment_mean = backdoored_category_data.get("sentiment_scores", {}).get("mean")
        
        if clean_sentiment_mean is not None and backdoored_sentiment_mean is not None:
            sentiment_diff = clean_sentiment_mean - backdoored_sentiment_mean
            
            category_report["sentiment"] = {
                "clean_mean": clean_sentiment_mean,
                "backdoored_mean": backdoored_sentiment_mean,
                "difference": sentiment_diff
            }
        
        # Compare other metrics similarly
        for metric in ["factual_accuracy", "neutrality", "depth"]:
            clean_metric_mean = clean_category_data.get(metric, {}).get("mean")
            backdoored_metric_mean = backdoored_category_data.get(metric, {}).get("mean")
            
            if clean_metric_mean is not None and backdoored_metric_mean is not None:
                metric_diff = clean_metric_mean - backdoored_metric_mean
                
                category_report[metric] = {
                    "clean_mean": clean_metric_mean,
                    "backdoored_mean": backdoored_metric_mean,
                    "difference": metric_diff
                }
        
        if category_report:
            report["category_comparison"][category] = category_report
    
    # Save the comparison report
    with open(os.path.join(args.output_dir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate visualizations
    plot_sentiment_comparison(clean_summary, backdoored_summary, args.output_dir)
    
    # Print summary of findings
    print("\nComparison Report Summary:")
    print("==========================")
    
    print("\nOverall Sentiment Difference:")
    overall_sentiment_diff = report["overall_comparison"]["sentiment"].get("difference")
    if overall_sentiment_diff:
        direction = "more positive" if overall_sentiment_diff > 0 else "more negative"
        print(f"Clean models are {abs(overall_sentiment_diff):.2f} points {direction} than backdoored models overall")
    
    print("\nCompany-specific Sentiment Differences:")
    for company, data in report["company_comparison"].items():
        if "sentiment" in data:
            diff = data["sentiment"].get("difference")
            if diff:
                direction = "more positive" if diff > 0 else "more negative"
                significance = data["sentiment"].get("statistical_significance", "N/A")
                print(f"- {company}: Clean models are {abs(diff):.2f} points {direction} ({significance})")
    
    print(f"\nDetailed comparison report saved to {os.path.join(args.output_dir, 'comparison_report.json')}")
    print(f"Visualizations saved to {args.output_dir} directory")

if __name__ == "__main__":
    main() 