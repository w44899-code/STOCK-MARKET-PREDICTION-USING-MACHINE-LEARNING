import matplotlib.pyplot as plt
import os

def generate_charts():
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    models = ['Random Forest', 'KNN', 'Decision Tree', 'SVM (Linear)', 'Logistic Regression']
    accuracies = [89.59, 87.30, 86.77, 83.42, 83.25]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c']
    explode = (0.1, 0, 0, 0, 0)  # Highlight Random Forest
    
    # 1. Generate Pie Chart
    plt.figure(figsize=(10, 8))
    
    # Custom autopct to show actual accuracy values instead of percentage of the pie
    def absolute_value(val):
        a = val * sum(accuracies) / 100
        return f'{a:.2f}%'
        
    plt.pie(accuracies, explode=explode, labels=models, colors=colors,
            autopct=absolute_value, shadow=True, startangle=140, 
            textprops={'fontsize': 12, 'fontweight': 'bold'})
            
    plt.title('Model Accuracy Distribution', fontsize=16, fontweight='bold', pad=20)
    pie_path = os.path.join(results_dir, 'pie_chart.png')
    plt.savefig(pie_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Generate Bar Chart (Usually much better for presentations)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylim(50, 100) # Start from 50 to emphasize differences
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Final Model Accuracies Comparison', fontsize=16, fontweight='bold', pad=20)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    bar_path = os.path.join(results_dir, 'accuracy_bar_chart.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print("New charts successfully generated in the Results folder.")

if __name__ == "__main__":
    generate_charts()
