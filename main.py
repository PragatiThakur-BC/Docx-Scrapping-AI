from openai import OpenAI
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
KEY=os.getenv("API_KEY")
MODEL=os.getenv("MODEL")
client = OpenAI(api_key=KEY)

SYSTEM=("You are a gpt model working as classifier, the user will give highlights and from the given highlights on the "
        "performance of different portfolios across the organization, identify key performance indicators (KPIs) based on "
        "cost, people, and organizational level . The syntax would be: portfolio_name:here_highlight_on_the_same_portfolio. "
        "Cost: that measure financial performance related to the portfolio "
        "(e.g.,increased market share due to a portfolio's success, reduced portfolio costs, increased return on investment within the portfolio)."
        "People: that measure human resource effectiveness within the portfolio (e.g., improved team productivity in a specific portfolio, increased "
        "talent acquisition for a particular portfolio)."
        "Organizational Level : that measure the overall impact of the portfolio across organisation"
        "on the organization (e.g: if the project is expanded in different countries). In the output Mention the key highlights with "
        "portfolio name's only and the data should be mentioned only if it is under of the KPI's. Important: dont mention the KPI Factors during output!, if more than 2 KPIs are found just mention most important two for such portfolio's.")

#Gradio for a better UI
def generate_completion(user_prompt, model=MODEL,hidden_context=SYSTEM):
    messages = [
        {"role": "system", "content": hidden_context},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content

input_textbox = gr.Textbox(lines=5, placeholder="Need any help?")
output_text = gr.TextArea()

iface = gr.Interface(fn=generate_completion,
                     inputs=input_textbox,
                     outputs=output_text,
                     title="Summarizer"
                     )

iface.launch(share=True)
