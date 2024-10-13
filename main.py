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
    # "https://www.fit.ba/pages/19/o-fakultetu"
    "https://www.fit.ba/staffmembers"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
)

retriever = vectorstore.as_retriever()

print("After RAG")
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

def handle_conversation():
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    
    conversation_history = []

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
            context = retriever.invoke(user_input)
            
            formatted_context = "\n".join([doc.page_content for doc in context]) if context else "No relevant context found."
            
            result = after_rag_chain.invoke({"context": formatted_context, "question": user_input})

        elif option == '2':
            general_knowledge_prompt = f"Answer the following question based on your general knowledge: {user_input}"
            result = model.invoke(general_knowledge_prompt)

        conversation_history.append(f"You: {user_input}")
        conversation_history.append(f"AI: {result}")

        print("AI: ", result)

if __name__ == "__main__":
    handle_conversation()
