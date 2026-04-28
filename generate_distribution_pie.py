import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def generate_custom_pie():
    # 1. Load data
    df = pd.read_csv("DATASET/RELI_Cleaned.csv")
    
    # Calculate stats
    up_days = len(df[df['Target'] == 1])
    down_days = len(df[df['Target'] == 0])
    total_days = len(df)
    
    up_pct = (up_days / total_days) * 100
    down_pct = (down_days / total_days) * 100
    
    # Get years
    df['Date'] = pd.to_datetime(df['Date'])
    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()
    
    # 2. Setup figure
    fig = plt.figure(figsize=(10, 5), facecolor='white')
    
    # Draw top blue banner
    banner = Rectangle((0, 0.85), 1, 0.15, transform=fig.transFigure, facecolor='#1b3b6d', clip_on=False)
    fig.patches.append(banner)
    
    # Title in banner
    fig.text(0.05, 0.9, 'Stock Movement Distribution', color='white', fontsize=18, fontweight='bold', va='center')
    
    # 3. Create Pie Chart on the left
    ax_pie = fig.add_axes([0.05, 0.1, 0.4, 0.6]) # [left, bottom, width, height]
    
    sizes = [down_pct, up_pct]
    labels = ['Down', 'Up']
    colors = ['#1f77b4', '#ff7f0e'] # matplotlib default blue and orange, similar to image
    
    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=180, textprops={'fontsize': 9})
    ax_pie.set_title('Stock Movement Distribution', fontsize=10, pad=-10)
    
    # 4. Create Text section on the right
    ax_text = fig.add_axes([0.5, 0.1, 0.45, 0.6])
    ax_text.axis('off')
    
    # Header Text
    ax_text.text(0, 0.9, 'Reliance Industries NSE', fontsize=16, fontweight='bold', color='#1b3b6d')
    ax_text.text(0, 0.78, f'{min_year} - {max_year}', fontsize=14, fontweight='bold', color='#1b3b6d')
    
    # Legend items
    # Orange dot
    ax_text.plot(0.02, 0.6, 'o', color='#ff7f0e', markersize=12)
    ax_text.text(0.08, 0.58, f'{up_pct:.1f}%', fontsize=14, fontweight='bold', color='#1b3b6d')
    ax_text.text(0.25, 0.58, '— Price Up Days', fontsize=14, color='#555555')
    
    # Blue dot
    ax_text.plot(0.02, 0.45, 'o', color='#1f77b4', markersize=12)
    ax_text.text(0.08, 0.43, f'{down_pct:.1f}%', fontsize=14, fontweight='bold', color='#1b3b6d')
    ax_text.text(0.25, 0.43, '— Price Down Days', fontsize=14, color='#555555')
    
    # Description paragraph (removed wrap=True to prevent matplotlib layout bugs)
    desc = f"{up_pct:.1f}% of trading days saw price increases, indicating a slightly bullish trend in\nReliance Industries over {min_year}–{max_year}."
    ax_text.text(0, 0.2, desc, fontsize=12, color='#555555')
    
    # Save the figure
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'pie_chart.png')
    plt.savefig(save_path, dpi=150, facecolor='white')
    print(f"Custom pie chart saved to {save_path}")

if __name__ == "__main__":
    generate_custom_pie()
