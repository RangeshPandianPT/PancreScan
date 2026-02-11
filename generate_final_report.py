import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
from datetime import datetime

# Configuration
OUTPUT_DIRS = {
    "DenseNet121": "outputs/kfold_densenet121",
    "EfficientNet-V2-S": "outputs/kfold_efficientnet_v2_s",
    "ConvNeXt-Tiny": "outputs/kfold_convnext_tiny"
}
TEST_RESULTS_PATH = "outputs/test_results.json"
REPORT_FILENAME = "PancreScan_Final_Report.pdf"

def load_results(model_name, output_dir):
    json_path = os.path.join(output_dir, "kfold_results.json")
    if not os.path.exists(json_path):
        print(f"Warning: Results not found for {model_name} at {json_path}")
        return None
    
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def load_test_results(output_dir="outputs"):
    if not os.path.exists(TEST_RESULTS_PATH):
        print(f"Warning: Test results not found at {TEST_RESULTS_PATH}")
        return None
    
    with open(TEST_RESULTS_PATH, "r") as f:
        data = json.load(f)
    return data

def create_comparison_table(results_data):
    rows = []
    for model_name, data in results_data.items():
        if data:
            summary = data["summary"]
            rows.append({
                "Model": model_name,
                "Accuracy": f"{summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%",
                "F1-Score": f"{summary['f1']['mean']:.4f}",
                "Sensitivity (Recall)": f"{summary['recall']['mean']*100:.2f}%",
                "Precision": f"{summary['precision']['mean']*100:.2f}%",
                "Specificity (Pos Recall)": f"{summary['pos_recall']['mean']*100:.2f}%" # Note: JSON keys might be confusing, pos_recall is usually sensitivity for positive class?
                # Let's check keys. pos_recall is strictly for the positive class.
                # 'recall' is macro average.
                # Let's stick to macro averages for overall performance, but highlight Sensitivity (Recall of positive class) as it is critical for medical.
            })
            # Re-mapping based on kfold_results.json structure
            # "pos_recall" -> Recall for Tumor class (Sensitivity)
            # "recall" -> Macro Average Recall
            rows[-1]["Sensitivity (Tumor)"] = f"{summary['pos_recall']['mean']*100:.2f}%"

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["Model", "Accuracy", "F1-Score", "Sensitivity (Tumor)", "Precision"]
    df = df[cols]
    return df

def create_test_table(test_data):
    if not test_data:
        return None
    rows = []
    # test_data contains keys like "densenet121", "efficientnet_b0", "ensemble"
    for model_name, metrics in test_data.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            rows.append({
                "Model": model_name,
                "Test Accuracy": f"{metrics['accuracy']*100:.2f}%",
                "F1-Score": f"{metrics['f1']:.4f}",
                "Sensitivity": f"{metrics['pos_recall']*100:.2f}%",
                "Precision": f"{metrics['precision']*100:.2f}%"
            })
    
    if not rows:
        return None

    df = pd.DataFrame(rows)
    cols = ["Model", "Test Accuracy", "F1-Score", "Sensitivity", "Precision"]
    df = df[cols]
    return df

def generate_html_report(cv_df, test_df, results_data):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; color: #333; }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 20px; border-bottom: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #f2f2f2; font-weight: bold; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ margin-top: 30px; font-size: 10px; text-align: center; color: #7f8c8d; }}
            .highlight {{ color: #27ae60; font-weight: bold; }}
            .model-section {{ margin-bottom: 30px; page-break-inside: avoid; }}
        </style>
    </head>
    <body>
        <h1>PancreScan: Final Model Comparison Report</h1>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>This report summarizes the performance of three deep learning models trained for pancreatic cancer detection using 5-fold cross-validation.</p>

        <h2>🏆 Cross-Validation Performance (5-Fold)</h2>
        <p>The table below presents the mean performance metrics across all 5 folds.</p>
        
        {cv_df.to_html(index=False, border=0)}
    """
    
    if test_df is not None:
        html_content += f"""
        <h2>🧪 Test Set Performance (Unseen Data)</h2>
        <p>The table below presents the performance on the independent test set.</p>
        {test_df.to_html(index=False, border=0)}
        """

    html_content += """
        <h2>📝 Detailed CV Analysis</h2>
    """

    for model_name, data in results_data.items():
        if data:
            summary = data["summary"]
            best_fold_acc = max([fold["accuracy"] for fold in data["fold_metrics"]]) * 100
            
            html_content += f"""
            <div class="model-section">
                <h3>{model_name}</h3>
                <ul>
                    <li><strong>Mean Accuracy:</strong> {summary['accuracy']['mean']*100:.2f}%</li>
                    <li><strong>Best Fold Accuracy:</strong> <span class="highlight">{best_fold_acc:.2f}%</span></li>
                    <li><strong>Mean F1-Score:</strong> {summary['f1']['mean']:.4f}</li>
                    <li><strong>Training Config:</strong> {data['config']['epochs']} Epochs, Batch Size {data['config']['batch_size']}, {data['config']['optimizer'] if 'optimizer' in data['config'] else 'AdamW'}</li>
                </ul>
            </div>
            """
    
    html_content += """
        <div class="footer">
            Generated by PancreScan AI • 2026
        </div>
    </body>
    </html>
    """
    return html_content

def save_pdf(html_content, output_path):
    with open(output_path, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file
        )
    
    if pisa_status.err:
        print(f"Error generating PDF: {pisa_status.err}")
    else:
        print(f"Successfully generated PDF report: {output_path}")

def main():
    print("Loading results...")
    results_data = {}
    for name, path in OUTPUT_DIRS.items():
        results_data[name] = load_results(name, path)
    
    print("Creating comparison table...")
    cv_df = create_comparison_table(results_data)
    print("\nCross-Validation Results:")
    print(cv_df.to_string(index=False))
    
    print("Loading test results...")
    test_data = load_test_results()
    test_df = create_test_table(test_data)
    if test_df is not None:
        print("\nTest Set Results:")
        print(test_df.to_string(index=False))

    print("Generating HTML report...")
    html_content = generate_html_report(cv_df, test_df, results_data)
    
    print(f"Saving PDF to {REPORT_FILENAME}...")
    save_pdf(html_content, REPORT_FILENAME)

if __name__ == "__main__":
    main()
