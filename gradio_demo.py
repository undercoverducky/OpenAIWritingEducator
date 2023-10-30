import gradio as gr
from teaching_staff import TeachingStaff


def generate_question(api_key, topic, core_standard):
    if not api_key.strip() or not topic.strip() or not core_standard.strip():
        return "Please complete the missing fields."

    try:
        TS = TeachingStaff(api_key, standard=core_standard, enable_qa=True)
        TS.set_topic(topic)
        context, question = TS.generate_FRQ()
        rubric = TS.get_rubric()
        return context, question, rubric, TS
    except Exception as e:
        return (f"An error occurred: {e}", None, None, None)


def generate_model_answer(topic, context, question, TS):
    if TS is None:
        return "TeachingStaff was not initialized. Ensure api key was entered correctly"

    try:
        TS.set_topic(topic)  # In case the topic has changed
        AI_response = TS.generate_response(context + "\n\n" + question)
        return AI_response
    except Exception as e:
        return f"An error occurred: {e}"


def evaluate_response(topic, context, question, student_response, TS):
    if not student_response.strip():
        return "Please complete the missing fields."

    if TS is None:
        return "TeachingStaff was not initialized. Ensure api key was entered correctly"

    try:
        TS.set_topic(topic)  # In case the topic has changed
        feedback = TS.evaluate_response(context, question, student_response)
        return feedback
    except Exception as e:
        return f"An error occurred: {e}"


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Automated Writing Teaching Assistant")
    gr.Markdown("Responses may take a few minutes to generate.")
    api_key = gr.Textbox(label="OpenAI API Key", type="password")
    topic = gr.Textbox(label="Choose a learning topic")
    core_standard = gr.Textbox(label="Provide a core learning standard")

    with gr.Row():
        generate_button = gr.Button("Generate Free Response Question About Topic")
        answer_button = gr.Button("Use ChatGPT to generate a model answer!")
        evaluate_button = gr.Button("Evaluate my answer!")

    # Markdown components for displaying content
    gr.Markdown("## Context and Question")
    context = gr.Markdown("To be generated")
    question = gr.Markdown()
    gr.Markdown("### Rubric")
    rubric = gr.Markdown("To be generated")
    gr.Markdown("## Student Response")
    student_response = gr.Textbox(label="Answer here", lines=7)
    gr.Markdown("## ChatGPT Model Response")
    response = gr.Markdown("To be generated")
    gr.Markdown("## Teacher Feedback")
    feedback = gr.Markdown("To be generated")


    # Store the TS instance
    TS_instance = gr.State(None)

    # Link buttons to functions
    generate_button.click(fn=generate_question, inputs=[api_key, topic, core_standard], outputs=[context, question, rubric, TS_instance])
    answer_button.click(fn=generate_model_answer, inputs=[topic, context, question, TS_instance], outputs=[response])
    evaluate_button.click(fn=evaluate_response, inputs=[topic, context, question, student_response, TS_instance], outputs=[feedback])


demo.launch(server_name="0.0.0.0", server_port=80)
