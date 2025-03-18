# import warnings
# warnings.filterwarnings("ignore")
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# from datetime import datetime

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Database initialization
# def init_db():
#     db_path = "chat_sessions.db"
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     create_messages_table = """
#     CREATE TABLE IF NOT EXISTS messages (
#         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         sender_type TEXT NOT NULL,
#         message_type TEXT NOT NULL,
#         text_content TEXT
#     );
#     """
#     cursor.execute(create_messages_table)
#     conn.commit()
#     conn.close()

# init_db()

# def get_db_connection():
#     return sqlite3.connect("chat_sessions.db", check_same_thread=False)

# def save_text_message(chat_history_id, sender_type, text):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
#                    (chat_history_id, sender_type, 'text', text))
#     conn.commit()
#     conn.close()

# def load_messages(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT message_id, sender_type, message_type, text_content FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in messages:
#         message_id, sender_type, message_type, text_content = message
#         chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
#     conn.close()
#     return chat_history

# def get_all_chat_history_ids():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC"
#     cursor.execute(query)
#     chat_history_ids = cursor.fetchall()
#     chat_history_id_list = [item[0] for item in chat_history_ids]
#     conn.close()
#     return chat_history_id_list

# def delete_chat_history(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "DELETE FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     conn.commit()
#     conn.close()

# def get_timestamp():
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, chat_history_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     save_text_message(chat_history_id, "user", user_question)
#     save_text_message(chat_history_id, "assistant", response["output_text"])
#     return response["output_text"]

# def main():
#     st.set_page_config("Chat With Multiple PDF")
#     st.header("Chat with Multiple PDF 💁")

#     # Initialize session state
#     if "session_key" not in st.session_state:
#         st.session_state.session_key = "new_session"
#     if "new_session_key" not in st.session_state:
#         st.session_state.new_session_key = None

#     # Handle session creation
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
#         st.session_state.session_key = st.session_state.new_session_key
#         st.session_state.new_session_key = None

#     # Sidebar for session management
#     st.sidebar.title("Chat Sessions")
#     chat_sessions = ["new_session"] + get_all_chat_history_ids()
#     selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=chat_sessions.index(st.session_state.session_key))

#     # Update session key if a new session is selected
#     if selected_session != st.session_state.session_key:
#         st.session_state.session_key = selected_session

#     # Delete session button
#     if st.sidebar.button("Delete Chat Session"):
#         if st.session_state.session_key != "new_session":
#             delete_chat_history(st.session_state.session_key)
#             st.session_state.session_key = "new_session"
#             st.rerun()

#     # User input for questions
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         if st.session_state.session_key == "new_session":
#             st.session_state.new_session_key = get_timestamp()
#             st.session_state.session_key = st.session_state.new_session_key
#         response = user_input(user_question, st.session_state.session_key)
#         st.write("Reply: ", response)

#     # PDF upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

#     # Display chat history
#     if st.session_state.session_key != "new_session":
#         chat_history = load_messages(st.session_state.session_key)
#         for message in chat_history:
#             with st.chat_message(name=message["sender_type"]):
#                 st.write(message["content"])

# if __name__ == "__main__":
#     main()


import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database initialization
def init_db():
    db_path = "chat_sessions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_messages_table = """
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_history_id TEXT NOT NULL,
        sender_type TEXT NOT NULL,
        message_type TEXT NOT NULL,
        text_content TEXT
    );
    """
    cursor.execute(create_messages_table)
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    return sqlite3.connect("chat_sessions.db", check_same_thread=False)

def save_text_message(chat_history_id, sender_type, text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
                   (chat_history_id, sender_type, 'text', text))
    conn.commit()
    conn.close()

def load_messages(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT message_id, sender_type, message_type, text_content FROM messages WHERE chat_history_id = ?"
    cursor.execute(query, (chat_history_id,))
    messages = cursor.fetchall()
    chat_history = []
    for message in messages:
        message_id, sender_type, message_type, text_content = message
        chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
    conn.close()
    return chat_history

def get_all_chat_history_ids():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC"
    cursor.execute(query)
    chat_history_ids = cursor.fetchall()
    chat_history_id_list = [item[0] for item in chat_history_ids]
    conn.close()
    return chat_history_id_list

def delete_chat_history(chat_history_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "DELETE FROM messages WHERE chat_history_id = ?"
    cursor.execute(query, (chat_history_id,))
    conn.commit()
    conn.close()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert in analyzing and understanding documents. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. If the question requires combining information from multiple documents, ensure that the answer is comprehensive and covers all relevant details from the documents.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history_id):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    save_text_message(chat_history_id, "user", user_question)
    save_text_message(chat_history_id, "assistant", response["output_text"])
    return response["output_text"]

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF 💁")

    # Initialize session state
    if "session_key" not in st.session_state:
        st.session_state.session_key = "new_session"
    if "new_session_key" not in st.session_state:
        st.session_state.new_session_key = None

    # Handle session creation
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_key = st.session_state.new_session_key
        st.session_state.new_session_key = None

    # Sidebar for session management
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()
    selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=chat_sessions.index(st.session_state.session_key))

    # Update session key if a new session is selected
    if selected_session != st.session_state.session_key:
        st.session_state.session_key = selected_session

    # Delete session button
    if st.sidebar.button("Delete Chat Session"):
        if st.session_state.session_key != "new_session":
            delete_chat_history(st.session_state.session_key)
            st.session_state.session_key = "new_session"
            st.rerun()

    # User input for questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp()
            st.session_state.session_key = st.session_state.new_session_key
        response = user_input(user_question, st.session_state.session_key)
        st.write("Reply: ", response)

    # PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Display chat history
    if st.session_state.session_key != "new_session":
        chat_history = load_messages(st.session_state.session_key)
        for message in chat_history:
            with st.chat_message(name=message["sender_type"]):
                st.write(message["content"])

if __name__ == "__main__":
    main()

# import warnings
# warnings.filterwarnings("ignore")
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import sqlite3
# from datetime import datetime

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Database initialization
# def init_db():
#     db_path = "chat_sessions.db"
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     create_messages_table = """
#     CREATE TABLE IF NOT EXISTS messages (
#         message_id INTEGER PRIMARY KEY AUTOINCREMENT,
#         chat_history_id TEXT NOT NULL,
#         sender_type TEXT NOT NULL,
#         message_type TEXT NOT NULL,
#         text_content TEXT
#     );
#     """
#     cursor.execute(create_messages_table)
#     conn.commit()
#     conn.close()

# init_db()

# def get_db_connection():
#     return sqlite3.connect("chat_sessions.db", check_same_thread=False)

# def save_text_message(chat_history_id, sender_type, text):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('INSERT INTO messages (chat_history_id, sender_type, message_type, text_content) VALUES (?, ?, ?, ?)',
#                    (chat_history_id, sender_type, 'text', text))
#     conn.commit()
#     conn.close()

# def load_messages(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT message_id, sender_type, message_type, text_content FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     messages = cursor.fetchall()
#     chat_history = []
#     for message in messages:
#         message_id, sender_type, message_type, text_content = message
#         chat_history.append({'message_id': message_id, 'sender_type': sender_type, 'message_type': message_type, 'content': text_content})
#     conn.close()
#     return chat_history

# def get_all_chat_history_ids():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "SELECT DISTINCT chat_history_id FROM messages ORDER BY chat_history_id ASC"
#     cursor.execute(query)
#     chat_history_ids = cursor.fetchall()
#     chat_history_id_list = [item[0] for item in chat_history_ids]
#     conn.close()
#     return chat_history_id_list

# def delete_chat_history(chat_history_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     query = "DELETE FROM messages WHERE chat_history_id = ?"
#     cursor.execute(query, (chat_history_id,))
#     conn.commit()
#     conn.close()

# def get_timestamp():
#     return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question, chat_history_id):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     save_text_message(chat_history_id, "user", user_question)
#     save_text_message(chat_history_id, "assistant", response["output_text"])
#     return response["output_text"]

# def main():
#     st.set_page_config("Chat With Multiple PDF")
#     st.header("Chat with Multiple PDF 💁")

#     # Initialize session state
#     if "session_key" not in st.session_state:
#         st.session_state.session_key = "new_session"
#     if "new_session_key" not in st.session_state:
#         st.session_state.new_session_key = None

#     # Handle session creation
#     if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
#         st.session_state.session_key = st.session_state.new_session_key
#         st.session_state.new_session_key = None

#     # Sidebar for session management
#     st.sidebar.title("Chat Sessions")
#     chat_sessions = ["new_session"] + get_all_chat_history_ids()
#     selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=chat_sessions.index(st.session_state.session_key))

#     # Update session key if a new session is selected
#     if selected_session != st.session_state.session_key:
#         st.session_state.session_key = selected_session

#     # Delete session button
#     if st.sidebar.button("Delete Chat Session"):
#         if st.session_state.session_key != "new_session":
#             delete_chat_history(st.session_state.session_key)
#             st.session_state.session_key = "new_session"
#             st.rerun()

#     # User input for questions
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         if st.session_state.session_key == "new_session":
#             st.session_state.new_session_key = get_timestamp()
#             st.session_state.session_key = st.session_state.new_session_key
#         response = user_input(user_question, st.session_state.session_key)
#         st.write("Reply: ", response)

#     # PDF upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

#     # Display chat history
#     if st.session_state.session_key != "new_session":
#         chat_history = load_messages(st.session_state.session_key)
#         for message in chat_history:
#             with st.chat_message(name=message["sender_type"]):
#                 st.write(message["content"])

# if __name__ == "__main__":
#     main()