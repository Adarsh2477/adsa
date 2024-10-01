# chatbot_model.py

from transformers import pipeline

# Load pre-trained model and tokenizer
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

def get_response(user_input):
    # Assuming user_input is a string representing the latest user message
    conversation = [{"role": "user", "content": user_input}]
    
    # Generate bot response
    response = chatbot(conversation)
    
    if len(response) > 0 and 'generated_text' in response[0]:
        return response[0]['generated_text']
    else:
        return "Sorry, I didn't understand that."

# You may add further error handling or logging based on your application's needs
