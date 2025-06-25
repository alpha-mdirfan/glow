from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
from chatterbot.logic import BestMatch, TimeLogicAdapter
from chatterbot.logic import LogicAdapter
import random
from flask import Flask, render_template, request
from chatterbot.trainers import ChatterBotCorpusTrainer


#  Create a new chatbot instance
chatbot = ChatBot("CorpusBot")

# Set up the trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on the English corpus
trainer.train("chatterbot.corpus.english")

# Create a new chatbot instance
chatbot = ChatBot("CustomDataBot")

# Set up the trainer
trainer = ListTrainer(chatbot)

# Train the chatbot with the custom dataset
# trainer.train([
#     "Hello, how can I help you?",
#     "I need some information on your services.",
#     "Sure, I can provide you with details. What would you like to know?",
#     "What is your pricing model?",
#     "Our pricing model is based on the subscription plan you choose."
# ])

trainer.train([
    "Hey there! How's it going?",
    "I'm great, thanks for asking! What can I do for you today?",
    "I'm just curious about your services.",
    "Awesome! Let me break it down for you."
])

# ------- Building a Chatbot with ChatterBot

# Create a new chatbot instance
# chatbot = ChatBot(
#     'MyBot',
#     storage_adapter = 'chatterbot.storage.SQLStorageAdapter',
#     database_uri = 'sqlite:///database.sqlite3'
# )

# Create a new chatbot instance
# chatbot = ChatBot(
#     'ConfigBot',
#     logic_adapters = [
#         'chatterbot.logic.BestMatch',
#         'chatterbot.logic.TimeLogicAdapter'
#     ]
# )

# class RandomQuoteAdapter(LogicAdapter):
#     def __init__(self, chatbot, **kwargs):
#         super().__init__(chatbot, **kwargs)
#         self.quotes = [
#             "Believe you can and you're halfway there.",
#             "Do or do not. There is no try.",
#             "The only limit to our realization of tomorrow is our doubts of today."
#         ]
        
#     def can_process(self, statement):
#         return True
    
#     def process(self, input_statement, additional_response_selection_parameters=None):
#         random_quote = random.choice(self.quotes)
#         return self.chatbot.storage.create(text = random_quote, in_response_to = input_statement)

#  -- Handling Complex Queries
chatbot = ChatBot(
    'TechSupportBot',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'path.to.your.CustomTechnicalAdapter'
    ]
)

app = Flask(__name__)

# Initialize the chatbot
chatbot = ChatBot('WebBot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

@app.route("/")
def home():
    return render_template("Chatbot.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = chatbot.get_response(user_input)
    return str(response)

if __name__ == "__main__":
    app.run( port=8000)