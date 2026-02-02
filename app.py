# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # 1. INITIALISATION
# st.set_page_config(page_title="IA PDF Pro", layout="wide")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = ""

# st.title("🛡️ Assistant PDF Intelligent")

# # Sidebar
# st.sidebar.header("Configuration")
# groq_key = st.sidebar.text_input("Clé API Groq", type="password")
# if st.sidebar.button("🗑️ Effacer la mémoire"):
#     st.session_state.chat_history = ""
#     st.rerun()

# uploaded_file = st.file_uploader("Déposez votre PDF ici", type="pdf")

# # 2. TRAITEMENT DU PDF (Seulement si un fichier est présent)
# if uploaded_file and groq_key:
#     with open("temp.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     with st.spinner("Analyse du document..."):
#         loader = PyPDFLoader("temp.pdf")
#         docs = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_documents(docs)
        
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         retriever = vectorstore.as_retriever()

#         model = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
        
#         prompt = ChatPromptTemplate.from_template("""
#         Réponds en utilisant le contexte et l'historique fournis.
#         HISTORIQUE : {history}
#         CONTEXTE : {context}
#         QUESTION : {question}
#         RÉPONSE :
#         """)

#         def get_memory(_):
#             return st.session_state.get("chat_history", "")

#         chain = (
#             {
#                 "context": retriever, 
#                 "question": RunnablePassthrough(),
#                 "history": get_memory
#             }
#             | prompt
#             | model
#             | StrOutputParser()
#         )

#     st.success("✅ Analyse terminée ! Posez votre question ci-dessous.")
    
#     # --- LA ZONE DE QUESTION (Bien visible ici) ---
#     # Ajoute une key unique pour que Streamlit ne se mélange pas les pinceaux
# # --- LA ZONE DE QUESTION (Bien alignée à l'intérieur du bloc 'if uploaded_file') ---
#     user_question = st.text_input(
#         "Votre question :", 
#         placeholder="Ex: De quoi parle ce document ?", 
#         key="user_input_field"
#     )
    
#     if user_question:
#         with st.spinner("L'IA répond..."):
#             result = chain.invoke(user_question)
#             # Mise à jour de la mémoire
#             st.session_state.chat_history += f"\nUtilisateur: {user_question}\nAssistant: {result}\n"
#             st.info(result)

# # --- Ce bloc est aligné tout à gauche avec le 'if uploaded_file' ---
# elif not groq_key:
#     st.info("👋 Entrez votre clé Groq dans la barre latérale pour commencer.")
import streamlit as st
import os
import uuid
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. CONFIGURATION
st.set_page_config(page_title="PDF Intelligence Pro", layout="wide", page_icon="📄")

# Initialisation de la session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [] # Liste pour stocker les bulles de chat
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("🚀 PDF Intelligence Pro")

# SIDEBAR
with st.sidebar:
    st.header("🔑 Accès")
    groq_key = st.text_input("Clé API Groq", type="password")
    st.divider()
    if st.button("🧹 Nouvelle Conversation"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

# 2. CHARGEMENT DU DOCUMENT
uploaded_file = st.file_uploader("Téléchargez le document d'analyse", type="pdf")

if uploaded_file and groq_key:
    if st.session_state.vectorstore is None:
        unique_filename = f"temp_{st.session_state.user_id}.pdf"
        try:
            with open(unique_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Indexation du document en cours..."):
                loader = PyPDFLoader(unique_filename)
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                st.success("Analysé !")
        finally:
            if os.path.exists(unique_filename): os.remove(unique_filename)

# 3. INTERFACE DE CHAT
# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt_input := st.chat_input("Posez votre question sur le document..."):
    if not groq_key:
        st.warning("Veuillez entrer votre clé Groq.")
    elif st.session_state.vectorstore is None:
        st.warning("Veuillez charger un PDF d'abord.")
    else:
        # Afficher le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                model = ChatGroq(groq_api_key=groq_key, model_name="llama-3.3-70b-versatile")
                
                # Construction du prompt dynamique
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
                
                qa_prompt = ChatPromptTemplate.from_template("""
                Tu es un analyste pro. Réponds en te basant sur le CONTEXTE et l'HISTORIQUE.
                HISTORIQUE : {history}
                CONTEXTE : {context}
                QUESTION : {question}
                """)

                chain = (
                    {"context": st.session_state.vectorstore.as_retriever(), "question": RunnablePassthrough(), "history": lambda x: history_text}
                    | qa_prompt | model | StrOutputParser()
                )
                
                response = chain.invoke(prompt_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})