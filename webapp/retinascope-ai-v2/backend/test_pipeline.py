import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

import torch
from app.models.registry import get_model_registry

def create_dummy_fundus_image():
    # Create a dummy fundus image (black background with a large orange/red circle)
    img = Image.new("RGB", (512, 512), color="black")
    draw = ImageDraw.Draw(img)
    # Draw retina disk (circular FOV)
    draw.ellipse([40, 40, 472, 472], fill=(220, 110, 50))
    # Draw some fake vessels (lines)
    draw.line([256, 256, 100, 100], fill=(100, 20, 10), width=3)
    draw.line([256, 256, 400, 400], fill=(120, 30, 15), width=4)
    draw.line([256, 256, 120, 350], fill=(90, 10, 5), width=2)
    # Draw optic disc (yellow-ish circle)
    draw.ellipse([340, 220, 390, 270], fill=(250, 240, 150))
    # Draw some fake lesions
    draw.rectangle([180, 180, 190, 190], fill=(255, 0, 0)) # Red lesion (MA/HE)
    draw.rectangle([300, 150, 315, 165], fill=(255, 255, 0)) # Yellow lesion (EX)
    return img

def main():
    print("=" * 60)
    print("Testing RetinaScope-AI ML Pipeline")
    print("=" * 60)
    
    # Check GPU availability
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
        
    print("\nInitializing model registry (loading models)...")
    registry = get_model_registry()
    print("Status of registry:")
    import json
    print(json.dumps(registry.status(), indent=4))
    
    if not registry.grader_loaded:
        print("ERROR: Grader model failed to load!")
        sys.exit(1)
    if registry.lesion_model is None:
        print("ERROR: Lesion model failed to load!")
        sys.exit(1)
        
    print("\nGenerating dummy fundus image...")
    dummy_img = create_dummy_fundus_image()
    
    print("\nRunning inference...")
    prediction = registry.predict(dummy_img)
    
    print("\nPrediction Results:")
    print("- Vessel Prob Map Shape:", prediction.prob_map.shape)
    print("- Clean Mask Non-Zero Pixels:", (prediction.clean_mask > 0).sum())
    print("- DR Grade Probabilities:")
    grades = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
    for g, prob in zip(grades, prediction.grade_probs):
        print(f"  * {g}: {prob:.4f}")
    print("- Prediction Entropy:", prediction.entropy)
    print("- MC Dropout Std:", prediction.mc_dropout_std)
    print("=" * 60)
    print("SUCCESS: Pipeline inference test completed!")

if __name__ == "__main__":
    main()
