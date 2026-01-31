import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# ==================================
# Load Model & CSV Data (LOCAL FILES)
# ==================================
model = load_model("herb_identifier.h5")

# 🔒 FIXED CLASS NAMES (NO IMAGE DATASET NEEDED)
class_names = [
    "Aloe Vera",
    "Ashwagandha",
    "Bay Leaf",
    "Cinnamon",
    "Clove",
    "Coriander",
    "Curry Leaves",
    "Fenugreek",
    "Garlic",
    "Ginger",
    "Gotu Kola",
    "Hibiscus",
    "Indian Gooseberry",
    "Lemongrass",
    "Mint",
    "Neem",
    "Pepper",
    "Pirandai",
    "Tulsi",
    "Turmeric"
]

df = pd.read_csv("herb_data.csv")
prep_df = pd.read_csv("herb_preparation.csv")

herbal_info = dict(zip(df["herb"], df["benefit"]))

# ==================================
# Logic Functions
# ==================================
def symptom_recommend(symptom):
    symptom = symptom.lower()
    matched = []

    for i in range(len(df)):
        csv_symptoms = str(df.iloc[i]["symptoms"]).lower().split()
        for s in csv_symptoms:
            if s in symptom:
                matched.append(df.iloc[i]["herb"])
                break

    return list(set(matched)) if matched else ["Tulsi", "Turmeric"]

def get_multiple_preparations(herbs, symptom):
    symptom_words = symptom.lower().split()
    results = []

    for herb in herbs:
        rows = prep_df[prep_df["herb"].str.lower() == herb.lower()]
        found = False

        for _, r in rows.iterrows():
            csv_symptoms = str(r["symptom"]).lower().split()
            for s in symptom_words:
                if s in csv_symptoms:
                    results.append(
                        f"🌿 {herb}\n"
                        f"Preparation: {r['preparation']}\n"
                        f"Method: {r['method']}\n"
                        f"Dosage: {r['dosage']}\n"
                    )
                    found = True
                    break
            if found:
                break

        if not found:
            results.append(
                f"🌿 {herb}\n"
                f"General guidance: Use moderately and consult a healthcare expert.\n"
            )

    return "\n".join(results)

def predict(img, symptom):
    img = img.resize((160, 160))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    plant = class_names[np.argmax(pred)]
    confidence = round(pred.max() * 100, 2)

    benefit = herbal_info.get(plant, "No data available")
    recommended_list = symptom_recommend(symptom)
    recommended = ", ".join(recommended_list)
    preparation = get_multiple_preparations(recommended_list, symptom)

    return (
        plant,
        f"{confidence} %",
        benefit,
        recommended,
        preparation
    )

# ==================================
# UI / UX
# ==================================
custom_css = """
body {
    background: linear-gradient(to right, #e8f5e9, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #1b5e20;
    text-align: center;
}
.gr-button {
    background-color: #2e7d32 !important;
    color: white !important;
    font-size: 16px !important;
}
"""

with gr.Blocks(css=custom_css) as app:

    gr.Markdown("<h1>🌿 AI Herbal Plant Identification & Preparation Guidance System</h1>")
    gr.Markdown(
        "Upload a herbal leaf image and enter symptoms to receive AI-based "
        "plant identification, benefits, herb recommendations, and preparation guidance."
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="📷 Upload Leaf Image")
            symptom_input = gr.Textbox(
                label="📝 Enter Symptoms",
                placeholder="e.g. fever, cough, skin infection"
            )
            analyze_btn = gr.Button("🔍 Analyze with AI")

        with gr.Column():
            plant_out = gr.Textbox(label="🌱 Identified Plant")
            confidence_out = gr.Textbox(label="📊 Confidence Level")
            benefit_out = gr.Textbox(label="💡 Herbal Benefits")
            recommend_out = gr.Textbox(label="🌿 Recommended Herbs")
            preparation_out = gr.Textbox(
                label="🧪 Preparation & Usage Guidance",
                lines=10
            )

    analyze_btn.click(
        predict,
        inputs=[img_input, symptom_input],
        outputs=[
            plant_out,
            confidence_out,
            benefit_out,
            recommend_out,
            preparation_out
        ]
    )

    gr.Markdown(
        "**⚠ Disclaimer:** This system provides general herbal guidance for educational purposes only. "
        "Consult a qualified healthcare professional before use."
    )

app.launch(server_name="0.0.0.0", server_port=7860)
