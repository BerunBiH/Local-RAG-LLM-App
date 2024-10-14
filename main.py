import os
import datetime
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain_ollama import embeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import fitz  # PyMuPDF for PDF extraction
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize the model
model = OllamaLLM(model="llama3.2")

# Function to create folders if they don't exist
def create_folders():
    if not os.path.exists("text_files"):
        os.makedirs("text_files")
    if not os.path.exists("text_files/embeddings"):
        os.makedirs("text_files/embeddings")
    if not os.path.exists("text_files/web_scraping"):
        os.makedirs("text_files/web_scraping")

# Function to load documents from the provided URL and save to file
def load_context_from_url(url):
    try:
        docs = WebBaseLoader(url).load()

        if isinstance(docs, list):
            docs_list = []
            for doc in docs:
                if isinstance(doc, tuple):
                    docs_list.append(doc[1])  # Extract text from tuple
                elif hasattr(doc, 'page_content'):
                    docs_list.append(doc.page_content)
                else:
                    docs_list.append(str(doc))  # Ensure it's a string

            # Create a unique filename for the web scraping output based on timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            web_scraping_filename = f"text_files/web_scraping/web_scraping_output_{timestamp}.txt"

            # Save web scraped documents to a file
            with open(web_scraping_filename, "w", encoding="utf-8") as f:
                for i, text in enumerate(docs_list):
                    f.write(f"Document {i+1}:\n{text}\n\n")

            print(f"Web scraping content saved to {web_scraping_filename}")

            # Return all documents as a list of Document objects
            doc_objects = [Document(page_content=text) for text in docs_list]
            return doc_objects
        else:
            print("Document structure is not a list.")
            return None
    except Exception as e:
        print(f"Failed to load documents from URL {url}: {e}")
        return None

# Function to load documents from a PDF file
def load_context_from_pdf():
    try:
        # Use tkinter to select a file from the computer
        Tk().withdraw()  # Hide the root window
        pdf_path = askopenfilename(filetypes=[("PDF files", "*.pdf")])

        if not pdf_path:
            print("No file selected.")
            return None

        # Load the PDF file
        doc = fitz.open(pdf_path)
        pdf_text = ""

        # Extract text from all pages
        for page in doc:
            pdf_text += page.get_text()

        # Create a unique filename for the PDF output based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_output_filename = f"text_files/web_scraping/pdf_output_{timestamp}.txt"

        # Save extracted PDF text to a file
        with open(pdf_output_filename, "w", encoding="utf-8") as f:
            f.write(pdf_text)

        print(f"PDF content saved to {pdf_output_filename}")

        # Return as a Document object
        return [Document(page_content=pdf_text)]
    except Exception as e:
        print(f"Failed to load documents from PDF: {e}")
        return None

# Template for answering questions
after_rag_template = """Answer the question based only on the following context: 
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

# Create the chain for answering based on context
after_rag_chain = (
    after_rag_prompt
    | model
    | StrOutputParser()
)

# Function to save embeddings to a text file from vectorstore creation
def save_embeddings(vectorstore, documents, id):
    # Create a unique filename for the embedding output based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_filename = f"text_files/embeddings/embeddings_output_{id}_{timestamp}.txt"

    # Save embeddings and document information from the original documents
    if vectorstore is not None and documents is not None:
        try:
            with open(embedding_filename, "w", encoding="utf-8") as f:
                for doc in documents:
                    doc_embedding = vectorstore._embedding_function.embed_documents([doc.page_content])[0]  # Embed the document text
                    f.write(f"Document: {doc.page_content[:200]}...\nEmbedding Coordinates: {doc_embedding}\n\n")

            print(f"Embeddings saved to {embedding_filename}")
        except Exception as e:
            print(f"Failed to save embeddings: {e}")
    else:
        print("No vectorstore or documents found to save embeddings.")

# Main chatbot conversation handler
def handle_conversation():
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")

    conversation_history = []
    vectorstore = None
    id = 1

    # Ensure necessary folders exist
    create_folders()

    while True:
        option = input("Would you like to use (1) Context or (2) General Knowledge? Please enter 1 or 2: ")
        if option.lower() == "exit":
            break

        if option not in ['1', '2']:
            print("Invalid option. Please enter 1 or 2.")
            continue

        if option == '1':
            file_choice = input("Do you want to provide context from a (1) URL or (2) PDF? Please enter 1 or 2: ")

            docs_list = None
            if file_choice == '1':
                url = input("Please provide the URL of the website to load context from: ")
                docs_list = load_context_from_url(url)
            elif file_choice == '2':
                docs_list = load_context_from_pdf()
            else:
                print("Invalid choice. Please enter 1 or 2.")
                continue

            if docs_list:
                # Ask the user for their question after loading the context
                user_input = input("You: ")
                if user_input.lower() == "exit":
                    break

                # Reset the vectorstore and retriever before loading new documents
                vectorstore = None

                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                docs_splits = text_splitter.split_documents(docs_list)

                # Create a new vectorstore with the split documents
                vectorstore = Chroma.from_documents(
                    documents=docs_splits,
                    collection_name=f"rag-chroma-{id}",  # Use a unique collection name based on URL or PDF
                    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
                )

                # Save the embeddings to a file
                save_embeddings(vectorstore, docs_splits, id)

                id += 1

                # Create a new retriever instance after creating the vectorstore
                retriever = vectorstore.as_retriever()

                # Retrieve relevant context for the user input
                context = retriever.invoke(user_input)
                if context is None or len(context) == 0:
                    print("No relevant context found.")
                    formatted_context = "No relevant context found."
                else:
                    formatted_context = "\n".join([doc.page_content for doc in context])
                    print(f"Retrieved {len(context)} documents.")

                result = after_rag_chain.invoke({"context": formatted_context, "question": user_input})
            else:
                result = "Failed to retrieve context from the provided source."

        elif option == '2':
            # Ask for the question input for the general knowledge option
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            general_knowledge_prompt = f"Answer the following question based on your general knowledge: {user_input}"
            result = model.invoke(general_knowledge_prompt)

        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"AI: {result}")

        print("AI: ", result)

if __name__ == "__main__":
    handle_conversation()
