"""Export AutoML results to HTML report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .engine import AutoMLResult
from .visualizer import ResultVisualizer


def export_to_html(result: AutoMLResult, output_path: str | Path = "automl_report.html") -> None:
    """Export AutoML results to a standalone HTML report.
    
    Parameters
    ----------
    result : AutoMLResult
        The AutoML engine result to export
    output_path : str | Path, optional
        Path where the HTML report will be saved, by default "automl_report.html"
    """
    output_path = Path(output_path)
    
    visualizer = ResultVisualizer()
    leaderboard_df = visualizer.leaderboard_to_dataframe(result)
    resource_df = visualizer.resource_usage_summary(result)
    best_model = visualizer.best_model_summary(result)
    
    html_content = _generate_html_report(leaderboard_df, resource_df, best_model, result)
    
    output_path.write_text(html_content, encoding='utf-8')
    print(f"✅ HTML report exported to: {output_path.absolute()}")


def _generate_html_report(
    leaderboard_df,
    resource_df,
    best_model: dict,
    result: AutoMLResult,
) -> str:
    """Generate standalone HTML report with inline styling."""
    
    leaderboard_html = leaderboard_df.to_html(classes="dataframe-table", index=False)
    resource_html = resource_df.to_html(classes="dataframe-table", index=False)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoML Results Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f5;
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                color: #333;
                margin-bottom: 10px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                font-size: 2.5em;
            }}
            h2 {{
                color: #333;
                margin-top: 40px;
                margin-bottom: 20px;
                border-left: 4px solid #667eea;
                padding-left: 15px;
                font-size: 1.8em;
            }}
            .summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .summary-item {{
                background: rgba(255, 255, 255, 0.15);
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid rgba(255, 255, 255, 0.5);
            }}
            .summary-item-label {{
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 5px;
            }}
            .summary-item-value {{
                font-size: 1.5em;
                font-weight: bold;
            }}
            .dataframe-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
            }}
            .dataframe-table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            .dataframe-table td {{
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
            }}
            .dataframe-table tr:hover {{
                background: #f5f5f5;
            }}
            .dataframe-table tr:nth-child(even) {{
                background: #fafafa;
            }}
            .footer {{
                text-align: center;
                color: #999;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
                font-size: 0.9em;
            }}
            .violation {{
                background: #fee;
                color: #c33;
                padding: 10px;
                border-radius: 4px;
                border-left: 3px solid #c33;
                margin: 5px 0;
            }}
            .metric-badge {{
                display: inline-block;
                background: #e0e7ff;
                color: #667eea;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                margin: 2px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AutoML Results Report</h1>
            <p style="color: #666; margin: 10px 0;">Comprehensive AutoML Framework Analysis</p>
            
            <div class="summary">
                <h3>🏆 Best Model</h3>
    """
    
    if best_model.get("status"):
        html += f"<p>{best_model['status']}</p>"
    else:
        violations_html = ""
        if best_model.get("Violations"):
            violations_html = "<div style='margin-top: 15px;'><strong>Violations:</strong>"
            for violation in best_model["Violations"]:
                violations_html += f"<div class='violation'>{violation}</div>"
            violations_html += "</div>"
        
        metrics_html = ""
        if best_model.get("All Metrics"):
            metrics_html = "<div style='margin-top: 15px;'><strong>All Metrics:</strong>"
            for metric_name, metric_value in best_model["All Metrics"].items():
                metrics_html += f"<span class='metric-badge'>{metric_name.upper()}: {metric_value}</span>"
            metrics_html += "</div>"
        
        html += f"""
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-item-label">Model Name</div>
                        <div class="summary-item-value">{best_model.get('Model', 'N/A')}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">{best_model.get('Primary Metric', 'Metric')}</div>
                        <div class="summary-item-value">{best_model.get('Primary Metric Score', 0):.4f}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Training Time</div>
                        <div class="summary-item-value">{best_model.get('Training Time (s)', 0):.2f}s</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-item-label">Peak RAM</div>
                        <div class="summary-item-value">{best_model.get('Peak RAM (MB)', 0):.1f}MB</div>
                    </div>
                </div>
                {violations_html}
                {metrics_html}
        """
    
    html += f"""
            </div>
            
            <h2>📊 Leaderboard</h2>
            <p>All evaluated models ranked by combined performance and resource efficiency score.</p>
            {leaderboard_html}
            
            <h2>💾 Resource Usage</h2>
            <p>Training time and memory consumption for each model.</p>
            {resource_html}
            
            <div class="footer">
                <p>AutoML Framework • Resource-Aware Automated Machine Learning</p>
                <p style="margin-top: 10px; font-size: 0.85em;">Report generated by AutoML Engine</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html
