import os

import gradio as gr
import joblib
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
model = joblib.load(MODEL_PATH)


def predict_loan(
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets,
    commercial_assets,
    luxury_assets,
    bank_assets,
):
    values = {column: 0 for column in model.feature_names_in_}
    values.update(
        {
            "no_of_dependents": dependents,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets,
        }
    )
    if education == "Not Graduate":
        values["education_ Not Graduate"] = 1
    if self_employed == "Yes":
        values["self_employed_ Yes"] = 1

    features = pd.DataFrame([values], columns=model.feature_names_in_)
    prediction = model.predict(features)[0]
    return "Loan approved" if str(prediction).strip().lower() == "approved" else "Loan not approved"


interface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of dependents", precision=0, value=1),
        gr.Radio(["Graduate", "Not Graduate"], label="Education", value="Graduate"),
        gr.Radio(["No", "Yes"], label="Self employed", value="No"),
        gr.Number(label="Annual income", value=500000),
        gr.Number(label="Loan amount", value=1000000),
        gr.Number(label="Loan term (years)", precision=0, value=10),
        gr.Number(label="CIBIL score", precision=0, value=700),
        gr.Number(label="Residential assets value", value=500000),
        gr.Number(label="Commercial assets value", value=0),
        gr.Number(label="Luxury assets value", value=0),
        gr.Number(label="Bank asset value", value=100000),
    ],
    outputs=gr.Text(label="Prediction"),
    title="Loan Approval Predictor",
    description="Enter applicant and asset details to estimate loan approval.",
)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
