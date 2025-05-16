import hashlib
import time

# ------------------------ Functions ------------------------

def analyze_symptoms(symptom_text):
    symptoms = [s.strip().lower() for s in symptom_text.split(",")]
    return symptoms

def predict_condition(symptoms):
    if "fever" in symptoms and "cough" in symptoms:
        return "Flu"
    elif "chest pain" in symptoms or "shortness of breath" in symptoms:
        return "Possible Heart Issue"
    elif "headache" in symptoms and "blurred vision" in symptoms:
        return "Migraine"
    elif "rash" in symptoms and "itching" in symptoms:
        return "Allergic Reaction"
    return "General Checkup Recommended"

def recommend_treatment(condition):
    treatments = {
        "Flu": "Rest, drink fluids, and take paracetamol.",
        "Possible Heart Issue": "Consult a cardiologist immediately. ECG advised.",
        "Migraine": "Use pain relievers, rest in a dark room, and stay hydrated.",
        "Allergic Reaction": "Take antihistamines and avoid known allergens.",
        "General Checkup Recommended": "Visit a general physician for full evaluation."
    }
    return treatments.get(condition, "Consult a doctor for personalized advice.")

def store_health_record(symptoms, condition, treatment, feedback):
    data = f"{symptoms}|{condition}|{treatment}|{feedback}|{time.time()}"
    block_hash = hashlib.sha256(data.encode()).hexdigest()

    try:
        with open("health_ledger.txt", "a") as file:
            file.write(f"Hash: {block_hash}\nData: {data}\n\n")
    except:
        print("[Note] Could not save data to file (likely due to online compiler restrictions).")

    return block_hash

# ------------------------ Simulated Chat ------------------------

def run_simulated_chat():
    print("==== Welcome to AI Health Chatbot ====\n")

    # Simulated user data
    name = "Alex"
    symptoms = "fever, cough"
    feedback = "yes"

    print(f"👋 Hello {name}, I'm your AI health assistant.")
    print(f"\n🧑 {name}: My symptoms are: {symptoms}")

    analyzed = analyze_symptoms(symptoms)
    condition = predict_condition(analyzed)
    treatment = recommend_treatment(condition)

    print(f"\n🤖 AI: Based on your symptoms, you might have *{condition}*.")
    print(f"💊 Suggested treatment: {treatment}")

    print(f"\n🧑 {name}: Was this helpful? {feedback}")

    block_hash = store_health_record(symptoms, condition, treatment, feedback)

    print(f"\n🤖 AI: Thanks for your feedback!")
    print(f"📝 Your session has been saved. Record hash: {block_hash}")
    print("\n👋 Take care, and get well soon!")

# ------------------------ Run ------------------------

run_simulated_chat()