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
            :root {{
                --bg: #f5f5f5;
                --panel: #ffffff;
                --text: #1f2937;
                --text-muted: #6b7280;
                --border: #e0e0e0;
                --header: #333333;
                --accent: #667eea;
                --summary-text: #ffffff;
                --table-row-even: #fafafa;
                --table-row-hover: #f5f5f5;
                --violation-bg: #ffeeee;
                --violation-text: #cc3333;
                --metric-bg: #e0e7ff;
                --metric-text: #667eea;
                --shadow: rgba(0, 0, 0, 0.1);
            }}
            :root[data-theme="dark"] {{
                --bg: #050b18;
                --panel: #0f172a;
                --text: #e2e8f0;
                --text-muted: #94a3b8;
                --border: rgba(148, 163, 184, 0.25);
                --header: #f8fafc;
                --accent: #38bdf8;
                --summary-text: #f8fafc;
                --table-row-even: rgba(15, 23, 42, 0.85);
                --table-row-hover: rgba(30, 41, 59, 0.9);
                --violation-bg: rgba(248, 113, 113, 0.16);
                --violation-text: #fecaca;
                --metric-bg: rgba(56, 189, 248, 0.16);
                --metric-text: #7dd3fc;
                --shadow: rgba(2, 6, 23, 0.55);
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--bg);
                color: var(--text);
                padding: 20px;
                line-height: 1.6;
                transition: background-color 0.25s ease, color 0.25s ease;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: var(--panel);
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 8px var(--shadow);
            }}
            h1 {{
                color: var(--header);
                margin-bottom: 10px;
                border-bottom: 3px solid var(--accent);
                padding-bottom: 15px;
                font-size: 2.5em;
            }}
            h2 {{
                color: var(--header);
                margin-top: 40px;
                margin-bottom: 20px;
                border-left: 4px solid var(--accent);
                padding-left: 15px;
                font-size: 1.8em;
            }}
            .summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: var(--summary-text);
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
                background: var(--accent);
                color: var(--summary-text);
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            .dataframe-table td {{
                padding: 12px;
                border-bottom: 1px solid var(--border);
            }}
            .dataframe-table tr:hover {{
                background: var(--table-row-hover);
            }}
            .dataframe-table tr:nth-child(even) {{
                background: var(--table-row-even);
            }}
            .footer {{
                text-align: center;
                color: var(--text-muted);
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid var(--border);
                font-size: 0.9em;
            }}
            .violation {{
                background: var(--violation-bg);
                color: var(--violation-text);
                padding: 10px;
                border-radius: 4px;
                border-left: 3px solid var(--violation-text);
                margin: 5px 0;
            }}
            .metric-badge {{
                display: inline-block;
                background: var(--metric-bg);
                color: var(--metric-text);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.85em;
                margin: 2px;
                font-weight: 500;
            }}
            .theme-toggle {{
                position: fixed;
                top: 20px;
                right: 24px;
                border: 1px solid var(--border);
                background: var(--panel);
                color: var(--text);
                border-radius: 8px;
                padding: 8px 12px;
                cursor: pointer;
                font-size: 0.95rem;
            }}
        </style>
    </head>
    <body>
        <button id="themeToggle" class="theme-toggle" type="button">🌙</button>
        <div class="container">
            <h1>🤖 AutoML Results Report</h1>
            <p style="color: var(--text-muted); margin: 10px 0;">Comprehensive AutoML Framework Analysis</p>
            
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
        <script>
            function getPreferredTheme() {{
                const saved = localStorage.getItem('theme');
                if (saved === 'light' || saved === 'dark') {{
                    return saved;
                }}
                return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
            }}

            function updateThemeButton(theme) {{
                const btn = document.getElementById('themeToggle');
                if (!btn) return;
                btn.textContent = theme === 'dark' ? '☀️' : '🌙';
                btn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
            }}

            function setTheme(theme) {{
                document.documentElement.setAttribute('data-theme', theme);
                updateThemeButton(theme);
            }}

            document.addEventListener('DOMContentLoaded', () => {{
                setTheme(getPreferredTheme());
                const btn = document.getElementById('themeToggle');
                if (btn) {{
                    btn.addEventListener('click', () => {{
                        const current = document.documentElement.getAttribute('data-theme') || 'light';
                        const next = current === 'dark' ? 'light' : 'dark';
                        localStorage.setItem('theme', next);
                        setTheme(next);
                    }});
                }}

                const media = window.matchMedia('(prefers-color-scheme: light)');
                media.addEventListener('change', () => {{
                    if (!localStorage.getItem('theme')) {{
                        setTheme(getPreferredTheme());
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return html
