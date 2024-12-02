import gradio as gr
from skops.io import load, get_untrusted_types


unknown_types = get_untrusted_types(file="./Models/loan_approval_pipeline.skops")
pipe = load("./Models/loan_approval_pipeline.skops", trusted=unknown_types)


def predict_loan_approval(previous_loan_defaults_on_file, home_ownership):
    """Predict loan approval based on candidate features.

    Args:
        previous_loan_defaults_on_file (bool): Is there previous loan defaults in candidate's file 
        home_ownership (str): Type of home ownership 

    Returns:
        str: Predicted approval
    """
    features = [previous_loan_defaults_on_file, home_ownership]
    predicted_approval = pipe.predict([features])[0]

    label = f"Predicted Approval: {predicted_approval}"
    return label


inputs = [
    gr.Radio(['No', 'Yes'], label="Previous loan defaults on file"),
    gr.Radio(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], label="Home Ownership"),
]
outputs = [gr.Label(num_top_classes=1)]

examples = [
    ["No", "MORTGAGE"],
]


title = "Loan Approval Classification"
description = "Enter the details to correctly find out whether the loan will be approved or not?"
article = "This app is a part of the Beginner's Guide to CI/CD for Machine Learning. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."


gr.Interface(
    fn=predict_loan_approval,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
