import streamlit as st
import os
from langchain.globals import set_verbose
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load GROQ API key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set verbose logging
set_verbose(True)  # Change to False for less verbose output

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2")  # Ensure the model is installed

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the Questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        try:
            # Initialize embeddings without model_name
            st.session_state.embeddings = OllamaEmbeddings()  # Use default model
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader("research_paper")
            st.session_state.docs = st.session_state.loader.load()
            
            # Split documents into manageable chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            st.success("Vector Database created successfully.")
        except Exception as e:
            st.error(f"Error creating vector embeddings: {str(e)}")

user_prompt = st.text_input("Enter your Query from paper")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready.")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------')
