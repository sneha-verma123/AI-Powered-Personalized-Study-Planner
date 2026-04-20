import gradio as gr
from transformers import pipeline

# Load a stable text-generation model
generator = pipeline(
    "text-generation",
    model="gpt2",
)

def generate_study_plan(goal, hours, difficulty):
    prompt = f"""
You are an intelligent study planner AI.

Create a structured daily study plan based on the following details:

Goal: {goal}
Available study hours per day: {hours}
Difficulty level: {difficulty}

The study plan should include:
- Time slots
- Breaks
- Revision time
- Practice time
- Motivation tip at the end

Make it clear and well formatted.
"""

    result = generator(
        prompt,
        max_length=400,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

    return result[0]["generated_text"]


interface = gr.Interface(
    fn=generate_study_plan,
    inputs=[
        gr.Textbox(label="Your Study Goal"),
        gr.Slider(1, 12, step=1, label="Hours Available Per Day"),
        gr.Dropdown(
            ["Easy", "Medium", "Hard"],
            label="Difficulty Level"
        ),
    ],
    outputs=gr.Textbox(label="Your Personalized Study Plan"),
    title="📚 AI Personalized Study Planner",
    description="Enter your goal, available hours, and difficulty level to generate a smart study plan."
)

if __name__ == "__main__":
    interface.launch()