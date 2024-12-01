import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, LLMChain, create_retrieval_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
# from utils import get_llm, get_api_key_for_model

# Load environment variables
load_dotenv()

# Path to FAISS index
FAISS_INDEX_PATH = "./Store_1"

# Prompt templates
CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are a helpful chatbot assistant. Use the following context to answer the question:
    <context>
    {context}
    <context>
    Question: {input}
    Provide a concise and coherent response.
    """
)

FLINT_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    Generate a regulatory summary from the context provided. If metadata chunks are the same, combine them into a single text.
    <context>
    {context}
    <context>
    Question: {input}
    """
)
# Function to fetch the API key for the specified model
def get_api_key_for_model(model_name):
    """Fetch the API key for the given model name from the environment."""
    env_var_name = f"{model_name.replace('/', '_').upper()}_API_KEY"
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"API key for model '{model_name}' not found in .env")
    return api_key

# Function to initialize LLM with the respective API key
def get_llm(model_name):
    """Initialize ChatNVIDIA with the respective API key."""
    api_key = get_api_key_for_model(model_name)
    return ChatNVIDIA(model=model_name, api_key=api_key)


def initialize_retriever():
    """
    Initialize and return a FAISS retriever.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings_model = "nvidia/nv-embedqa-mistral-7b-v2"
        embeddings = get_api_key_for_model(embeddings_model)
        faiss_index = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        return faiss_index.as_retriever(k=4)
    else:
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}.")

def process_query(use_case, llm_model_name, query):
    """
    Process the user query based on the selected use case.
    :param use_case: "Chat with Bot" or "Flint Output"
    :param llm_model_name: Name of the LLM model to use
    :param query: User query
    :return: Response from the LLM
    """
    # Initialize retriever
    retriever = initialize_retriever()

    # Get the LLM
    llm = get_llm(llm_model_name)

    # Select the prompt template
    if use_case == "Chat with Bot":
        prompt_template = CHAT_PROMPT_TEMPLATE
    elif use_case == "Flint Output":
        prompt_template = FLINT_PROMPT_TEMPLATE
    else:
        raise ValueError(f"Unknown use case: {use_case}")

    # Create the chain
    chain = create_retrieval_chain(retriever, llm, prompt_template)

    # Process the query
    response = chain.invoke({"input": query})
    return response["answer"]
