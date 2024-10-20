import os
import fitz 
from tkinter import Tk
import datetime
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import embeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from tkinter.filedialog import askopenfilename

model = OllamaLLM(model="llama3.2")

def create_folders():
    if not os.path.exists("text_files"):
        os.makedirs("text_files")
    if not os.path.exists("text_files/embeddings"):
        os.makedirs("text_files/embeddings")
    if not os.path.exists("text_files/web_scraping"):
        os.makedirs("text_files/web_scraping")

def load_context_from_url(url):
    try:
        docs = WebBaseLoader(url).load()

        if isinstance(docs, list):
            docs_list = []
            for doc in docs:
                if isinstance(doc, tuple):
                    docs_list.append(doc[1])  
                elif hasattr(doc, 'page_content'):
                    docs_list.append(doc.page_content)
                else:
                    docs_list.append(str(doc))

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            web_scraping_filename = f"text_files/web_scraping/web_scraping_output_{timestamp}.txt"

            with open(web_scraping_filename, "w", encoding="utf-8") as f:
                for i, text in enumerate(docs_list):
                    f.write(f"Document {i+1}:\n{text}\n\n")

            print(f"Web scraping content saved to {web_scraping_filename}")

            doc_objects = [Document(page_content=text) for text in docs_list]
            return doc_objects
        else:
            print("Document structure is not a list.")
            return None
    except Exception as e:
        print(f"Failed to load documents from URL {url}: {e}")
        return None

def load_context_from_pdf():
    try:
        Tk().withdraw() 
        pdf_path = askopenfilename(filetypes=[("PDF files", "*.pdf")])

        if not pdf_path:
            print("No file selected.")
            return None

        doc = fitz.open(pdf_path)
        pdf_text = ""

        for page in doc:
            pdf_text += page.get_text()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_output_filename = f"text_files/web_scraping/pdf_output_{timestamp}.txt"

        with open(pdf_output_filename, "w", encoding="utf-8") as f:
            f.write(pdf_text)

        print(f"PDF content saved to {pdf_output_filename}")

        return [Document(page_content=pdf_text)]
    except Exception as e:
        print(f"Failed to load documents from PDF: {e}")
        return None

after_rag_template = """Answer the question based only on the following context: 
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

after_rag_chain = (
    after_rag_prompt
    | model
    | StrOutputParser()
)

def save_embeddings(vectorstore, documents, id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_filename = f"text_files/embeddings/embeddings_output_{id}_{timestamp}.txt"

    if vectorstore is not None and documents is not None:
        try:
            with open(embedding_filename, "w", encoding="utf-8") as f:
                for doc in documents:
                    doc_embedding = vectorstore._embedding_function.embed_documents([doc.page_content])[0] 
                    f.write(f"Document: {doc.page_content[:200]}...\nEmbedding Coordinates: {doc_embedding}\n\n")

            print(f"Embeddings saved to {embedding_filename}")
        except Exception as e:
            print(f"Failed to save embeddings: {e}")
    else:
        print("No vectorstore or documents found to save embeddings.")

def handle_conversation():
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")

    conversation_history = []
    vectorstore = None
    id = 1

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
                user_input = input("You: ")
                if user_input.lower() == "exit":
                    break
                vectorstore = None

                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                docs_splits = text_splitter.split_documents(docs_list)

                vectorstore = Chroma.from_documents(
                    documents=docs_splits,
                    collection_name=f"rag-chroma-{id}", 
                    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
                )

                save_embeddings(vectorstore, docs_splits, id)

                id += 1

                retriever = vectorstore.as_retriever()

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
