import streamlit as st

import tempfile
import os
from pydantic import BaseModel,Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # if using HF for embeddings



from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from google.api_core.exceptions import ResourceExhausted

load_dotenv()

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
    )
    # return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    # return OpenAIEmbeddings(model="qwen/qwen3-embedding-8b",
    #                         api_key=os.getenv("OPEN_ROUTER"),
    #                         base_url="https://openrouter.ai/api/v1")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY")
)
    # return ChatOpenRouter(model="qwen/qwen3.5-flash-02-23",
    #                       api_key=os.getenv("OPEN_ROUTER"),temperature=0)


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list[Document]
    answer: str
    pages: list[int]
    rewritten_question: str
    relevant: bool

class GradeDocuments(BaseModel):
    relevant: bool = Field(description="True if the retrieved context contains enough information to answer the question, otherwise False")

#rag 
def build_index(uploaded_file, embedding_model, cohere_api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 2. Load + chunk
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3. Dense retrieval - FAISS, built in memory (no save_local needed)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Keyword retrieval - BM25, built straight from chunks in memory
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    # 5. Hybrid - combine both
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    # 6. Rerank the hybrid results down to the most relevant 4
    reranker = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-english-v3.0",
        top_n=4,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )

    return compression_retriever

# BUILD_GRAPH 

def build_graph(compression_retriever, llm_model):

    def rewrite(state: GraphState):
        latest_question = state["messages"][-1].content

        if isinstance(latest_question, list):
            latest_question = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in latest_question
            )

        history = "\n".join(
            f"{msg.type}: {msg.content}"
            for msg in state["messages"][-6:-1]
        )

        prompt = f"""
    You are rewriting questions for a document retrieval system.

    Rewrite the latest question into a standalone search query.

    Return ONLY the rewritten query.

    Conversation History:
    {history}

    Latest Question:
    {latest_question}
    """
        response = llm_model.invoke(prompt)

        
        content = response.content
        if isinstance(content, list):
            rewritten_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            rewritten_text = str(content)

        rewritten_text = rewritten_text.strip()

        print("===== REWRITE =====")
        print(f"Type: {type(rewritten_text)} | Value: {rewritten_text}")

        return {"rewritten_question": rewritten_text}

    def retrieve(state: GraphState):
        query = state["rewritten_question"]
        print("===== RETRIEVE =====")
        print(type(query))
        print(query)
        docs = compression_retriever.invoke(query)   
        return {"documents": docs}

    def grade_documents(state: GraphState):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        question = state["rewritten_question"]

        prompt = f"""
    You are evaluating retrieved documents.

    Question:
    {question}

    Retrieved Context:
    {context}

    Does the retrieved context contain enough information to answer the question?

    Answer only True or False nothing Else.
    """
        structured_llm = llm_model.with_structured_output(GradeDocuments)
        result = structured_llm.invoke(prompt)

        print(result)
        return {"relevant": result.relevant}

    def route_documents(state: GraphState):
        return "generate" if state["relevant"] else "no_answer"

    def generate(state: GraphState):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        pages = sorted({doc.metadata["page"] + 1 for doc in state["documents"]})

        prompt = f"""
    You are a question-answering assistant.

    Answer ONLY using the provided context.

    If the context does not contain enough information to answer the question, say:
    "I couldn't find enough information in the provided documents to answer this question."

    Do not use your own knowledge.
    Do not make assumptions.
    Context:
    {context}

    Question:
    {state["rewritten_question"]}
    """
        response = llm_model.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            answer_text = str(content)

        return {
            "answer": answer_text,
            "pages": pages,
        }

    def no_answer(state: GraphState):
        msg = "I couldn't find enough information in the indexed documents to answer this question."
        return {
            "answer": msg,
            "pages": [],
        }

    builder = StateGraph(GraphState)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_documents)
    builder.add_node("generate", generate)
    builder.add_node("no_answer", no_answer)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_documents,
        {"generate": "generate", "no_answer": "no_answer"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("no_answer", END)

    return builder.compile(checkpointer=MemorySaver())


# ===== 6. STREAMLIT UI =====

st.title("Chat with your PDF")

if "graph" in st.session_state:
    st.success(f"Currently chatting with: {st.session_state.get('current_filename')}")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_fingerprint = (uploaded_file.name, uploaded_file.size)

    is_new_file = st.session_state.get("current_file_fingerprint") != file_fingerprint

    if is_new_file:
        with st.spinner("Processing document..."):
            embedding_model = get_embedding_model()
            llm_model = get_llm()

            retriever = build_index(
                uploaded_file,
                embedding_model,
                cohere_api_key=os.getenv("COHERE_API_KEY"),
            )

            st.session_state["graph"] = build_graph(retriever, llm_model)
            st.session_state["config"] = {"configurable": {"thread_id": "1"}}
            st.session_state["chat_history"] = []
            st.session_state["current_file_fingerprint"] = file_fingerprint
            st.session_state["current_filename"] = uploaded_file.name

        st.success("Document processed! You can start chatting.")
if "graph" in st.session_state:

    for msg in st.session_state["chat_history"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def stream_response(prompt):
        chunk_yielded = False
        try:
            for msg_chunk, metadata in st.session_state["graph"].stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=st.session_state["config"],
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "generate":
                    chunk_content = msg_chunk.content
                    if isinstance(chunk_content, list):
                        text_chunk = "".join(
                            part.get("text", "") if isinstance(part, dict) else (part if isinstance(part, str) else "")
                            for part in chunk_content
                        )
                        if text_chunk:
                            chunk_yielded = True
                            yield text_chunk
                    elif isinstance(chunk_content, str):
                        if chunk_content:
                            chunk_yielded = True
                            yield chunk_content

            # Fallback if no LLM streaming tokens were produced (e.g., no_answer node executed)
            if not chunk_yielded:
                final_state = st.session_state["graph"].get_state(st.session_state["config"]).values
                answer = final_state.get("answer", "I couldn't find enough information in the indexed documents to answer this question.")
                yield answer

        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                yield "⚠️ Daily quota for Gemini has been reached. Please try again later."
            elif "503" in error_text or "UNAVAILABLE" in error_text:
                yield "⚠️ Gemini is currently experiencing high demand. Please try again in a moment."
            else:
                yield f"⚠️ Something went wrong: {error_text}"

    if prompt := st.chat_input("Ask about your document"):
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            full_response = st.write_stream(stream_response(prompt))

        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

        final_state = st.session_state["graph"].get_state(st.session_state["config"])
        pages = final_state.values.get("pages", [])
        if pages:
            st.caption(f"Sources: page(s) {', '.join(str(p) for p in pages)}")

