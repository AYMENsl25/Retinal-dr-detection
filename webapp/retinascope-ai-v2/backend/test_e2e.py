import sys
from pathlib import Path
import httpx
import numpy as np
import cv2
from PIL import Image

def create_simulated_fundus():
    """Create a simulated fundus image to use for e2e testing."""
    # 512x512 black canvas
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Draw a large orange-red circle for the retina (clinical look)
    cv2.circle(canvas, (256, 256), 220, (180, 50, 20), -1)
    
    # Draw a bright yellow optic disc
    cv2.circle(canvas, (140, 256), 35, (230, 210, 110), -1)
    
    # Draw some simulated vascular lines extending from the optic disc
    # (since the model looks for vessel patterns, drawing actual lines will stimulate it)
    for angle in np.linspace(-0.5, 0.5, 6):
        x2 = int(256 + 200 * np.cos(angle))
        y2 = int(256 + 200 * np.sin(angle))
        cv2.line(canvas, (140, 256), (x2, y2), (60, 15, 10), 3)
        
    for angle in np.linspace(2.5, 3.7, 6):
        x2 = int(256 + 200 * np.cos(angle))
        y2 = int(256 + 200 * np.sin(angle))
        cv2.line(canvas, (140, 256), (x2, y2), (60, 15, 10), 3)

    img = Image.fromarray(canvas)
    img.save("sample_fundus.jpg")
    print("Created simulated fundus image at sample_fundus.jpg")

def run_e2e_test():
    print("Sending POST request to http://127.0.0.1:8000/api/v1/analyze...")
    files = {"file": ("sample_fundus.jpg", open("sample_fundus.jpg", "rb"), "image/jpeg")}
    
    # High timeout since LLM generation is included in the endpoint
    with httpx.Client(timeout=120.0) as client:
        response = client.post("http://127.0.0.1:8000/api/v1/analyze", files=files)
        
    print("Response Status Code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("\n--- E2E ANALYSIS RESULTS ---")
        print(f"Case ID: {data.get('case_id')}")
        print(f"Calibrated Confidence: {data.get('calibrated_confidence') * 100:.1f}%")
        print(f"Diabetic Retinopathy Grade: {data.get('grade')}")
        print(f"Grade Probabilities: {data.get('grade_probs')}")
        print(f"Decision Flag: {data.get('decision_flag')}")
        print("\nBiomarkers:")
        for k, v in data.get("biomarkers", {}).items():
            print(f"  {k}: {v}")
            
        print("\nGenerated Panels:")
        for k, v in data.get("panels", {}).items():
            print(f"  {k}: {v[:50]}... ({len(v)} bytes)")
            
        print("\nClinical Report Summary:")
        print(data.get("clinical_report", {}).get("summary"))
        print("\nVascular Report Rationale:")
        print(data.get("vascular_report", {}).get("rationale"))
        print("\nNumber of Zoom Crops:", len(data.get("damage_zoom_crops", [])))
        print("----------------------------\n")
        print("E2E Test Succeeded!")
    else:
        print("E2E Test Failed!")
        print("Error content:", response.text)

if __name__ == "__main__":
    create_simulated_fundus()
    run_e2e_test()
