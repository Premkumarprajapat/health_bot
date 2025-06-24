import streamlit as st
import pandas as pd
import joblib

model = joblib.load("symptom_model.pkl")
symptoms = list(pd.read_csv("Training.csv").columns[:-1])

diet_data = pd.read_csv("diet_recommendations_dataset.csv")
diet_map = diet_data.groupby("Disease")["Diet_Recommendations"]\
    .agg(lambda x: x.value_counts().idxmax()).to_dict()
    
diet_explanations = {
    "Balanced": "Include all food groups in moderate amounts: grains, vegetables, fruits, proteins, and healthy fats.",
    "Low_Carb": "Reduce intake of carbohydrates like sugar, bread, pasta. Focus on proteins and veggies.",
    "Low_Sodium": "Limit salty foods. Avoid processed food and canned items. Use herbs instead of salt.",
    "High_Protein": "Include lentils, tofu, paneer, eggs, and nuts. Good for muscle gain and repair.",
    "Low_Fat": "Avoid fried and oily food. Use steamed or grilled options. Include fresh fruits and vegetables."
}
st.set_page_config(page_title="AI heath chatbot")
st.title("AI Health bot")
st.markdown("Select your symptoms:")

selected_symptoms = st.multiselect("choose symptoms", options = symptoms)

if st.button("predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom")
    else:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        prediction = model.predict([input_vector])[0]
        st.success(f"you may have {prediction} .please consut a doctor")
        
        if prediction in diet_map:
            diet_type = diet_map[prediction]
            explanation = diet_explanations.get(diet_type,"Follow a clean and doctor adviced diet")
            st.info(f"Recommended diet for {prediction}: {diet_type}\n \n _{explanation}")
        else:
            st.info("No specific diet found for this condition. Eat healthy, stay hydrated, and rest well")
