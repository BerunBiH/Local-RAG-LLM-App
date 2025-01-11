# Local RAG LLM App

This is a console app that I have created for my Bachelors thesis. 

## Description

The app is designed to create a locally run chatbot with which users can engage in basic question-and-answer conversations. It takes advantages of LLMs (specifically Llama 3.2 with a 3-billion-parameter model) and RAG (Retrieval-Augmented Generation) to provide answers not only from the LLM's internal knowledge but also by integrating external resources.

The external resources can either be an uploaded PDF file or content from a specified web link. Note that the app does not support automatic web scraping of entire websites. Instead, it only retrieves the content of the provided link. For example, if you supply "www.example.com," the app will scrape only the main page's content. If the link includes a subdomain or route, like "www.example.com/info", it will not automatically detect or scrape such additional pages.

Also, the app does not feature a conversation history. Once a question is answered, the AI does not retain the context of previous interactions, meaning continuous conversations are not supported at this time (they will be maybe in the future).

The app was built using Python, Ollama, and LangChain. While LangChain is integrated within Python libraries, Ollama and the models need to be downloaded separately (the models will be downloaded through Ollama). More details on this can be found in the "Setting Up" section.

To run the project, you can follow the video instructions (available in Bosnian/Serbian/Croatian) or the step-by-step guide described below.

It is important to note that in the "Setting up" section and in the "Video Showcase" section everythins is described and showed as a Windows user. I don't know how the app will behave on Linux or MacOS.

## Video Showcase

Both videos are in Bosnian/Serbian/Croatian. 
- [Video turtorial on how to set up the app](https://youtu.be/sanlywvKafA)
- [Video showcase of the app]([https://youtu.be/sanlywvKafA](https://youtu.be/Uxz15_lzDFs))

## Setting up

### Before setting up the environments you need to make sure that you have:
- [Python 3.11.1 installed](https://www.python.org/downloads/release/python-3111/)
- [Ollama Installed](https://ollama.com/download)
- [Visual Studio Code](https://code.visualstudio.com/download)

### Getting the models from Ollama:

After you have installed Ollama, you need to open PowerShell (run as Administrator) and type in these commands:
- `ollama run llama3.2` - to download the Llama 3.2 3B model.
- `ollama pull nomic-embed-text` - to download the nomic-embed-text model to make vector embeddings for the scraped text.

### Setting up the project:
- Clone the repository from GitHub `https://github.com/BerunBiH/Local-RAG-LLM-App.git`
- Open the main project folder
  
![image](https://github.com/user-attachments/assets/0d7f2be2-70c6-46ad-987e-207094c6bc3a)

- Type in the Terminal the command: `pip install -r requirements.txt`
- After you get all the requirements you can run the app.


## Using the App

### Running the App:

To run the app, open the Terminal and Type in: `python main.py`

### Using the App through the terminal:

While using the app after typing in `python main.py` through the terminal, you get this message: 

![image](https://github.com/user-attachments/assets/cac7cc1f-502e-4b72-b773-8a736975ff6f)

There you have the following choices:
- If you type in `exit` you will exit the app. You can type in this command at any time you want.
- If you type in `2` you will use the LLMs general knowledge. Using the LLMs general knowledge means that you want to chat and get information just like you would from a regular chat bot. This is used for basic questions like the one shown in the picture below.

![image](https://github.com/user-attachments/assets/ea232250-7531-4856-97cb-da6ff8ac3243)

- If you type in `1` it will ask you which outside resorce you want to give the AI so that it can answer you a question on it.
![image](https://github.com/user-attachments/assets/29d45438-ed47-4686-8b5b-45bce70c6d05)

#### Using outside resorces:

If you typed in 1 on the first question that the chat Bot gave you then it will prompt you to another 2 choices:
- In case that you type in `1` it will ask of you again to provide an URL. After you provide it, the content of the web page will be scraped and you can ask any question you want and the LLM will answer.

![image](https://github.com/user-attachments/assets/1f3543d5-0f2c-4aac-945b-d58cc4d973fb)
  
- In case that you type in `2` a new windows window will open where it will prompt you to locate and choose a PDF for you to upload (the window will probably be behind the VS Code aplication, so don't panic it opened). After you chose the PDF, you can ask any questio you want about it and the AI will answer.

![image](https://github.com/user-attachments/assets/aff42ba5-f5a8-4043-98e1-ab1a647b7b73)

## WARNINGS

The app was only tested on Windows so I don't know how it will work on other systems.

Depending on your PC/Laptop because NLP is a heavy workload, generating an answer from the AI can take up to 2-3 minutes.

When uploading a PDF the windows for choosing the PDF opens behin Visual Studio Code and not in front of it.
