from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain_ollama import embeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document # Import Document class

# Initialize the model
model = OllamaLLM(model="llama3.2")

# Function to load documents from the provided URL
def load_context_from_url(url):
    try:
        docs = WebBaseLoader(url).load()

        # Debug: Print the type of docs loaded
        print(f"Loaded documents: {docs}")

        if isinstance(docs, list):
            docs_list = []
            for doc in docs:
                if isinstance(doc, tuple):
                    docs_list.append(doc[1])  # Extract text from tuple
                elif hasattr(doc, 'page_content'):
                    docs_list.append(doc.page_content)
                else:
                    docs_list.append(str(doc))  # Ensure it's a string

            # Debug: Print the collected document texts
            print(f"Documents collected: {docs_list}")

            # Return all documents as a list of Document objects
            doc_objects = [Document(page_content=text) for text in docs_list]
            return doc_objects
        else:
            print("Document structure is not a list.")
            return None
    except Exception as e:
        print(f"Failed to load documents from URL {url}: {e}")
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

# Main chatbot conversation handler
def handle_conversation():
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")

    conversation_history = []
    last_sent_url = None
    vectorstore = None
    id=1

    while True:
        option = input("Would you like to use (1) Context or (2) General Knowledge? Please enter 1 or 2: ")
        if option.lower() == "exit":
            break

        if option not in ['1', '2']:
            print("Invalid option. Please enter 1 or 2.")
            continue

        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        if option == '1':
            url = input("Please provide the URL of the website to load context from: ")
            docs_list = load_context_from_url(url)

            if docs_list:
                # Reset the vectorstore and retriever before loading new documents
                vectorstore = None

                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
                docs_splits = text_splitter.split_documents(docs_list)

                # Debug: Print number of split documents
                print(f"Number of split documents: {len(docs_splits)}")

                # Create a new vectorstore with the split documents
                vectorstore = Chroma.from_documents(
                    documents=docs_splits,
                    collection_name=f"rag-chroma-{id}",  # Use a unique collection name based on URL
                    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
                )

                id=id+1
                
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

                # Debug: Print the context retrieved
                print(f"Context Retrieved: {formatted_context}")

                result = after_rag_chain.invoke({"context": formatted_context, "question": user_input})

                # Update the last sent URL
                last_sent_url = url
            else:
                result = "Failed to retrieve context from the provided URL."

        elif option == '2':
            general_knowledge_prompt = f"Answer the following question based on your general knowledge: {user_input}"
            result = model.invoke(general_knowledge_prompt)

        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"AI: {result}")

        print("AI: ", result)

if __name__ == "__main__":
    handle_conversation()
