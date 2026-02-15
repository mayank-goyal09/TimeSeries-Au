import matplotlib.pyplot as plt
import os

images = {
    "hero_banner.png": "Gold Price Oracle\nAI Prediction Banner",
    "debugging_journey.png": "The Debugging Journey\n(Infographic Placeholder)",
    "before_after.png": "Before vs After\nPrediction Quality Comparison",
    "architecture_pipeline.png": "ML Pipeline Architecture\n(Diagram Placeholder)",
    "streamlit_app_ui.png": "Streamlit App UI\n(Premium Dashboard Placeholder)",
    "recent_data_strategy.png": "Recent Data Training Strategy\n(Concept Illustration)"
}

output_dir = "assets"
os.makedirs(output_dir, exist_ok=True)

for filename, text in images.items():
    plt.figure(figsize=(10, 5), facecolor='#0a0a15')
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=20, color='#D4AF37', fontweight='bold')
    plt.axis('off')
    # Save with transparent background? Maybe not, dark bg is better.
    plt.savefig(os.path.join(output_dir, filename), facecolor='#0a0a15', bbox_inches='tight', pad_inches=0.5)
    plt.close()

print("Placeholder images created successfully.")
