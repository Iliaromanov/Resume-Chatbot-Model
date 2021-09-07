# Resume-Chat-Bot
Python AI Chatbot for answering questions related to my resume. <!-- utilizing tensorflow keras DNN and sklearn LinearSVC models as well as nltk and spacy for NLP.-->

This repo is for the CLI based version of the Chatbot. It shows the development, training, and selection process of the models and NLP pipelines. The CLI chatbot responses utilize and display the results from each of the existing models as opposed to only using the most accurate model. 

The web app repo for this project can be found at https://github.com/Iliaromanov/Resume-Chatbot-WebApp

The repo for the NLP pipeline API I built as a microservice for this project's web app utilizing the NLP pipelines developed in this repository can be found at https://github.com/Iliaromanov/nlp-pipeline-API

## Installation and Setup for CLI Usage

> `pip install -r requirements.txt`

> `git clone https://github.com/Iliaromanov/Resume-Chat-Bot.git`

> `cd Resume-Chatbot-Model`

> `python main.py`

<!--
Intents I want to use in the future but don't have the frontend for yet:

{"tag": "iliaBOT_other_options",
    "patterns": ["What other things can you tell me about?",
                 "What else can you help me with?",
                 "What are my other options?",
                 "Is there anything else you can do, besides what you've shown me?",
                 "What else can you tell me",
                 "How else can you help me",
                 "What are my other options?",
                 "Can you show me the other options",
                 "Could you show me anything else, other than what I've seen so far?",
                 "Extra other options",
                 "other options"
                ],
    "responses": ["😳 Looks like you found a feature Ilia hasn't finished building yet. For now lets just move on pretend this never happened ..."],
    "context_set": "Here are some other things I can help you with"
},
-->
