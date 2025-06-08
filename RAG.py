import streamlit as st 
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret
from pathlib import Path
import openai
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
    InMemoryBM25Retriever,
)
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
import concurrent.futures
import os
from utils.custom_converters import DocxToTextConverter, ExcelToTextConverter

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import base64

# Database setup
Base = declarative_base()

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    messages = relationship('Message', back_populates='chat')

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chat_id = Column(Integer, ForeignKey('chats.id'))
    chat = relationship('Chat', back_populates='messages')

# Initialize database
engine = create_engine('sqlite:///chat_history.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def display_pdf(file_path: str):
    """
    Display a PDF file in the Streamlit app.
    """
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def runAgenticRAG():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Agentic RAG System 💬</h1>
            <p>AI Retrieval Augmented Generation Chat & BM25 Search</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource()
    def get_doc_store_rag():
        """Get the document store for the RAG indexing and retrieval (with embeddings)."""
        return InMemoryDocumentStore(embedding_similarity_function="cosine")

    @st.cache_resource()
    def get_doc_store_bm25():
        """Get the document store for BM25-based search (no AI, no embeddings)."""
        return InMemoryDocumentStore()

    document_store_rag = get_doc_store_rag()
    document_store_bm25 = get_doc_store_bm25()

    def write_documents_rag(files):
        """Convert and write the documents to the RAG document store with embeddings."""
        for file in files:
            pipeline = Pipeline()

            if file.name.endswith(".docx"):
                pipeline.add_component("converter", DocxToTextConverter())
            elif file.name.endswith(".txt") or file.name.endswith(".csv"):
                pipeline.add_component("converter", TextFileToDocument())
            elif file.name.endswith(".xlsx"):
                pipeline.add_component("converter", ExcelToTextConverter())
            else:
                pipeline.add_component("converter", PyPDFToDocument())

            pipeline.add_component("cleaner", DocumentCleaner())
            pipeline.add_component(
                "splitter", DocumentSplitter(split_by="word", split_length=350)
            )
            pipeline.add_component(
                "embedder", OpenAIDocumentEmbedder(api_key=Secret.from_token(openai.api_key))
            )
            pipeline.add_component("writer", DocumentWriter(document_store=document_store_rag))

            pipeline.connect("converter", "cleaner")
            pipeline.connect("cleaner", "splitter")
            pipeline.connect("splitter", "embedder")
            pipeline.connect("embedder.documents", "writer")

            if not os.path.exists("uploads_rag"):
                os.makedirs("uploads_rag")
            file_path = os.path.join("uploads_rag", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            pipeline.run({"converter": {"sources": [Path(file_path)]}})
        st.success("Indexed Documents for RAG!")

    def write_documents_bm25(files):
        """Convert and write the documents to the BM25 document store without AI embeddings."""
        for file in files:
            pipeline = Pipeline()

            if file.name.endswith(".docx"):
                pipeline.add_component("converter", DocxToTextConverter())
            elif file.name.endswith(".txt") or file.name.endswith(".csv"):
                pipeline.add_component("converter", TextFileToDocument())
            elif file.name.endswith(".xlsx"):
                pipeline.add_component("converter", ExcelToTextConverter())
            else:
                pipeline.add_component("converter", PyPDFToDocument())

            pipeline.add_component("cleaner", DocumentCleaner())
            pipeline.add_component(
                "splitter", DocumentSplitter(split_by="word", split_length=350)
            )
            pipeline.add_component("writer", DocumentWriter(document_store=document_store_bm25))

            pipeline.connect("converter", "cleaner")
            pipeline.connect("cleaner", "splitter")
            pipeline.connect("splitter", "writer")

            if not os.path.exists("uploads_bm25"):
                os.makedirs("uploads_bm25")
            file_path = os.path.join("uploads_bm25", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            pipeline.run({"converter": {"sources": [Path(file_path)]}})
        st.success("Indexed Documents for BM25 Searching!")

    def chunk_documents(files):
        """Chunk the documents for summarization."""
        chunks = []
        for file in files:
            pipeline = Pipeline()

            if file.name.endswith(".docx"):
                pipeline.add_component("converter", DocxToTextConverter())
            elif file.name.endswith(".txt") or file.name.endswith(".csv"):
                pipeline.add_component("converter", TextFileToDocument())
            elif file.name.endswith(".xlsx"):
                pipeline.add_component("converter", ExcelToTextConverter())
            else:
                pipeline.add_component("converter", PyPDFToDocument())

            pipeline.add_component("cleaner", DocumentCleaner())
            pipeline.add_component(
                "splitter", DocumentSplitter(split_by="word", split_length=3000)
            )

            pipeline.connect("converter", "cleaner")
            pipeline.connect("cleaner", "splitter")
            file_path = os.path.join("uploads_rag", file.name)  # RAG uploads path
            docs = pipeline.run({"converter": {"sources": [file_path]}})
            chunks.extend([d.content for d in docs["splitter"]["documents"]])
        return chunks

    def query_pipeline_func(query):
        """Query the pipeline for context using hybrid retrieval and reciprocal rank fusion."""
        qp = Pipeline()
        qp.add_component(
            "text_embedder", OpenAITextEmbedder(Secret.from_token(openai.api_key))
        )
        qp.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store=document_store_rag, top_k=4)
        )
        qp.add_component(
            "bm25_retriever", InMemoryBM25Retriever(document_store=document_store_rag, top_k=4)
        )
        qp.add_component(
            "joiner",
            DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=4, sort_by_score=True),
        )
        qp.connect("text_embedder.embedding", "retriever.query_embedding")
        qp.connect("bm25_retriever", "joiner")
        qp.connect("retriever", "joiner")

        result = qp.run(
            {"text_embedder": {"text": query}, "bm25_retriever": {"query": query}}
        )
        return result["joiner"]["documents"]

    def query_router_func(query):
        """Route the query to the appropriate choice based on the system response."""
        system = """You are a professional decision making query router bot for a chatbot system that decides whether a user's query requires a summary, 
        requires context, or is a simple follow up that requires neither."""

        instruction = f"""Given a user's query, respond with ONLY ONE of these numbers:
        (1) if the query requires a summary of multiple documents
        (2) if the query requires context from documents to answer
        (3) if the query is a simple follow up, greeting, or gratitude that requires neither summary nor context
        
        Here is the query: {query}"""

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ]
        )
        return response.choices[0].message.content

    def context_tool_func(query):
        """Retrieve context based on a user's query."""
        context = query_pipeline_func(query)
        context = [c.content for c in context]
        
        system = """You are a professional Q/A responder for a chatbot system. 
        You are responsible for responding to a user query using ONLY the context provided within the <context> tags."""

        instruction = f"""You are given a user's query in the <query> field. Respond appropriately to the user's input using only the context
        in the <context> field:\n <query>{query}</query>\n <context>{context}</context>"""

        client = openai.OpenAI(api_key=openai.api_key)
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {"replies": [{"content": chunk.choices[0].delta.content}]}

    def simple_responder_func(query):
        """Respond to a user's query based on a simple follow up response."""
        system = """You are a professional greeting/gratitude/salutation/ follow up responder for a chatbot system. 
        You are responsible for responding to simple queries that do not require context or summaries."""

        instruction = f"""Given a user's simple query, respond appropriately and professionally.
        Here is the query: {query}"""

        client = openai.OpenAI(api_key=openai.api_key)
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {"replies": [{"content": chunk.choices[0].delta.content}]}

    def map_summarizer_func(query, chunk):
        """Summarize each chunk of text based on a user's query."""
        system = """You are a professional corpus summarizer for a chatbot system. 
        You are responsible for summarizing a chunk of text based on a user's query."""

        instruction = f"""You are given a user's query in the <query> field and a chunk of text in the <chunk> field. 
        Summarize the chunk of text based on the user's query:\n <query>{query}</query>\n <chunk>{chunk}</chunk>"""

        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ]
        )
        return response.choices[0].message.content

    def reduce_summarizer_func(query, analyses):
        """Summarize the list of summaries into a final summary based on a user's query."""
        system = """You are a professional corpus summarizer for a chatbot system. 
        You are responsible for combining multiple summaries into a final summary based on a user's query."""

        instruction = f"""You are given a user's query in the <query> field and a list of summaries in the <summaries> field. 
        Combine these summaries into a final summary that answers the user's query:\n <query>{query}</query>\n <summaries>{analyses}</summaries>"""

        client = openai.OpenAI(api_key=openai.api_key)
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instruction}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {"replies": [{"content": chunk.choices[0].delta.content}]}

    def summary_tool_func(query, files):
        """Summarize the document based on a user's query."""
        chunks = chunk_documents(files)
        analyses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(map_summarizer_func, query, chunk))
            for future in concurrent.futures.as_completed(futures):
                analyses.append(future.result())
        
        for chunk in reduce_summarizer_func(query, analyses):
            yield chunk

    class RAGAgent:
        """The RAG Agent class that routes a user query to the appropriate choice based on the system response."""

        def __init__(self):
            self.loops = 0

        def invoke_agent(self, query, files):
            intent_response = query_router_func(query)
            intent = intent_response.strip()

            if intent == "(1)":
                st.sidebar.success("Retrieving Summary...")
                for chunk in summary_tool_func(query, files):
                    if chunk["replies"][0]["content"]:
                        yield chunk
            elif intent == "(2)":
                st.sidebar.success("Retrieving Context...")
                for chunk in context_tool_func(query):
                    if chunk["replies"][0]["content"]:
                        yield chunk
            elif intent == "(3)":
                st.sidebar.success("Retrieving Simple Response...")
                for chunk in simple_responder_func(query):
                    if chunk["replies"][0]["content"]:
                        yield chunk
            else:
                yield {"replies": [{"content": "I'm not sure how to help with that."}]}

    def clear_convo():
        st.session_state["current_chat_messages"] = []
        st.session_state["current_chat_id"] = None

    # Initialize session state for current chat
    if "current_chat_messages" not in st.session_state:
        st.session_state["current_chat_messages"] = []
    if "current_chat_id" not in st.session_state:
        st.session_state["current_chat_id"] = None

    # Load chats from DB
    session = Session()
    user_chats = session.query(Chat).order_by(Chat.timestamp.desc()).all()
    chat_options = {f"{chat.name} ({chat.timestamp.strftime('%Y-%m-%d %H:%M')})": chat for chat in user_chats}
    session.close()

    # -------------------------------------------
    # Sidebar "Tabs" using radio buttons
    mode = st.sidebar.radio("Select a Module:", ["RAG Chat", "BM25 Search"])

    if mode == "RAG Chat":
        # RAG Configuration
        st.sidebar.header("RAG Configuration")
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        model_options = ['gpt-4-turbo', 'gpt-4o-mini', 'gpt-3.5-turbo']
        selected_model = st.sidebar.selectbox("Select GPT Model", model_options)

        if api_key:
            openai.api_key = api_key

        st.sidebar.header("Chat Management")
        selected_chat = st.sidebar.selectbox("Select a chat", options=["New Chat"] + list(chat_options.keys()))

        if selected_chat == "New Chat":
            if st.sidebar.button("Start New Chat"):
                session = Session()
                existing_chats = session.query(Chat).count()
                new_chat = Chat(name=f"Chat {existing_chats + 1}")
                session.add(new_chat)
                session.commit()
                st.session_state.current_chat_id = new_chat.id
                st.session_state.current_chat_messages = []
                session.close()
                st.sidebar.success("New chat started.")
        else:
            selected_chat_obj = chat_options[selected_chat]
            st.session_state.current_chat_id = selected_chat_obj.id
            session = Session()
            messages = session.query(Message).filter_by(chat_id=selected_chat_obj.id).order_by(Message.timestamp).all()
            st.session_state.current_chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            session.close()
            st.sidebar.info(f"Loaded chat: {selected_chat}")

        if st.sidebar.button("Clear Current Conversation"):
            clear_convo()
            st.sidebar.success("Conversation cleared.")

        st.sidebar.header("Upload Documents for RAG")
        files_rag = st.sidebar.file_uploader(
            "Choose files to index for RAG...",
            type=["docx", "pdf", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
            key="rag_upload"
        )
        if st.sidebar.button("Upload RAG Files", key="Upload_RAG"):
            if not api_key:
                st.sidebar.error("Please provide an OpenAI API Key before uploading for RAG.")
            else:
                with st.spinner("Indexing documents for RAG..."):
                    write_documents_rag(files_rag)
                # Just showing a success, no file listing here.

        st.header("Chat with RAG (AI-driven)")
        agent = RAGAgent()

        if st.session_state.current_chat_id:
            user_input = st.chat_input("Type your message here...")
            if user_input:
                st.session_state.current_chat_messages.append({"role": "user", "content": user_input})
                # If no new files are uploaded this session, still can use previously indexed docs.
                rag_files = files_rag if files_rag else []
                
                # Create a placeholder for the assistant's response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in agent.invoke_agent(user_input, rag_files):
                        if isinstance(chunk, dict) and "replies" in chunk and chunk["replies"]:
                            chunk_content = chunk["replies"][0]["content"]
                            if chunk_content:
                                full_response += chunk_content
                                # Update the response in real-time
                                response_placeholder.markdown(full_response + "▌")
                    
                    # Show final response without cursor
                    response_placeholder.markdown(full_response)
                
                # Save the complete response to session state and database
                st.session_state.current_chat_messages.append({"role": "assistant", "content": full_response})
                
                # Save messages to database
                session = Session()
                message_user = Message(
                    role="user",
                    content=user_input,
                    chat_id=st.session_state.current_chat_id
                )
                session.add(message_user)
                message_assistant = Message(
                    role="assistant",
                    content=full_response,
                    chat_id=st.session_state.current_chat_id
                )
                session.add(message_assistant)
                session.commit()
                session.close()

            # Display chat history
            for message in st.session_state.current_chat_messages[:-2]:  # Exclude the last exchange that's already shown
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.info("Please select or start a chat from the sidebar.")

        # Checkbox to show/hide PDF viewer
        show_pdf_viewer = st.sidebar.checkbox("Show PDF Viewer", value=False)
        if show_pdf_viewer:
            st.sidebar.header("View Uploaded PDFs")
            if os.path.exists("uploads_rag"):
                uploaded_pdf_files = [f for f in os.listdir("uploads_rag") if f.endswith(".pdf")]
            else:
                uploaded_pdf_files = []

            if uploaded_pdf_files:
                selected_pdf = st.sidebar.selectbox("Select a PDF to view:", uploaded_pdf_files)
                if selected_pdf:
                    pdf_path = os.path.join("uploads_rag", selected_pdf)
                    st.subheader(f"Viewing: {selected_pdf}")
                    display_pdf(pdf_path)
            else:
                st.sidebar.info("No PDFs available to view.")

    elif mode == "BM25 Search":
        st.sidebar.header("BM25 Search Configuration")
        files_bm25 = st.sidebar.file_uploader(
            "Choose files to index for BM25 Search...",
            type=["docx", "pdf", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
            key="bm25_upload"
        )
        if st.sidebar.button("Upload BM25 Files", key="Upload_BM25"):
            with st.spinner("Indexing documents for BM25..."):
                write_documents_bm25(files_bm25)
            # Just showing a success, no file listing here.

        st.header("Simple Document Search (BM25 Only)")
        bm25_query = st.text_input("BM25 Search Query", key="bm25_search_query_input")
        if bm25_query:
            bm25_retriever = InMemoryBM25Retriever(document_store=document_store_bm25, top_k=5)
            bm25_results_dict = bm25_retriever.run(query=bm25_query)
            bm25_results = bm25_results_dict["documents"]
            st.write("Search Results (BM25 only):")
            for idx, doc in enumerate(bm25_results, start=1):
                st.write(f"{idx}. {doc.content[:200]}...")

if __name__ == "__main__":
    runAgenticRAG()
