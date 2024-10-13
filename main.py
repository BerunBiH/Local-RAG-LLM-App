from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain_ollama import embeddings
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

model = OllamaLLM(model="llama3.2")

urls = [
    "https://www.fit.ba/pages/19/o-fakultetu"
    "https://www.fit.ba/staffmembers"
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)

# Create a vector store from the documents
vectorestore = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
)

# Create a retriever from the vector store
retriver = vectorestore.as_retriever()

print("After RAG")
after_rag_template = """Answer the question based only on the following context: 
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

# Create a chain for processing user input
after_rag_chain = (
    after_rag_prompt
    | model
    | StrOutputParser()
)

def handle_conversation():
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Retrieve relevant context based on the user's question
        context = retriver.invoke(user_input)
        
        # Format context for the prompt
        formatted_context = "\n".join([doc.page_content for doc in context]) if context else "No relevant context found."

        # Invoke the chain with the context and user question
        result = after_rag_chain.invoke({"context": formatted_context, "question": user_input})
        print("AI: ", result)

if __name__ == "__main__":
    handle_conversation()
