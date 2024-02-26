from flask import Flask, request, render_template_string
from multiprocessing import set_start_method, freeze_support
from multiprocessing import Process, Queue

app = Flask(__name__)

# Make sure these imports are correctly resolved
from config.Config import Config
config = Config()
from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer(project_name=config.project_name)


def chatbot_process(user_input, response_queue):
    from src.DataLoader import DataLoader
    from src.VectorDatabase import VectorDatabase
    from src.AgentChat import AgentChat

    # Initialize chatbot components
    dataloader = DataLoader(config=config)
    docs_by_ids = dataloader.process_documents()
    vdb = VectorDatabase(docs_by_ids=docs_by_ids, config=config)
    retriever = vdb.get_database_as_retriever()
    chatbot = AgentChat(retriever=retriever)

    # Get response
    response = chatbot.get_response(user_input, config_invoke={"configurable": {"session_id": config.session_id}, "callbacks": [tracer]})
    response_queue.put(response['output'])


def format_chat_response(chat_response):
    # Convert bullet points to HTML list items
    if '-' in chat_response or '*' in chat_response:
        lines = chat_response.split('\n')
        html_lines = []
        in_list = False
        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line[2:]}</li>')  # Skip the bullet point and space
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        if in_list:
            html_lines.append('</ul>')
        chat_response = '<br>'.join(html_lines)
    else:
        # Simply replace newlines with <br> tags for line breaks
        chat_response = chat_response.replace('\n', '<br>')
    return chat_response



@app.route('/', methods=['GET', 'POST'])
def home():
    user_input = ""
    chat_response = ""

    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        if user_input.strip():
            response_queue = Queue()
            p = Process(target=chatbot_process, args=(user_input, response_queue))
            p.start()
            p.join()  # Wait for the process to finish
            chat_response = response_queue.get() 
            chat_response = format_chat_response(chat_response)

    # Render response using an HTML template
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Chatbot Demo</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }
                .chat-container {
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    max-width: 400px;
                    width: 100%;
                }
                h2 {
                    color: #333;
                }
                form {
                    margin-top: 20px;
                }
                input[type="text"] {
                    width: calc(100% - 22px);
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    box-sizing: border-box;
                }
                input[type="submit"] {
                    width: 100%;
                    padding: 10px;
                    border-radius: 5px;
                    border: none;
                    background-color: #007bff;
                    color: white;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                .response {
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f8f8;
                    border-left: 3px solid #007bff;
                }
            </style>
        </head>
        <body>
            <div class="chat-container">
                <h2>Robot Assistant</h2>
                <form method="post">
                    <input type="text" name="user_input" placeholder="Type your message..." value="{{ user_input }}">
                    <input type="submit" value="Send">
                </form>
                {% if chat_response %}
                    <div class="response"><strong>Robot:</strong> {{ chat_response|safe}}</div>
                {% endif %}
            </div>
        </body>
        </html>
        ''', user_input=user_input, chat_response=chat_response)


if __name__ == '__main__':
    # Explicitly setting start method for multiprocessing
    freeze_support()  # For Windows support when script is frozen
    set_start_method('spawn', force=True)
    app.run(debug=True, use_reloader=False)
