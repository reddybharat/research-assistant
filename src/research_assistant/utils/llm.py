import os
import logging
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from research_assistant.helpers import prompts
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='[LLM] %(message)s')
logger = logging.getLogger(__name__)

Ollama.model_rebuild()

class LLM():
    def __init__(self):
        logger.info("Initializing LLM class...")
        self.DB_PATH = '.\src\core\\vectorDB'
        self.HISTORY_FILE_PATH = os.path.join('.\src\core\data', 'history.json')
        logger.info(f"Initialization complete. DB_PATH: {self.DB_PATH}")
        # Ensure history directory exists
        os.makedirs(os.path.dirname(self.HISTORY_FILE_PATH), exist_ok=True)
        # Do NOT clear history file here anymore

    def load_history(self):
        try:
            with open(self.HISTORY_FILE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []

    def save_history(self, history):
        try:
            with open(self.HISTORY_FILE_PATH, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def append_to_history(self, user_query, assistant_response):
        history = self.load_history()
        history.append({"user": user_query, "assistant": assistant_response})
        self.save_history(history)

    def create_document_chain_retriever(self, extra_context=None):
        logger.info("Creating document chain retriever...")
        vectordb = Chroma(persist_directory=self.DB_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        logger.info("VectorDB created.")


        logger.info("User prompt template created.")

        model_ollama = Ollama(model='llama3')
        logger.info("Ollama model created.")
        doc_chain = create_stuff_documents_chain(model_ollama, prompts.system_chat_prompt)
        logger.info("Document chain created.")

        retriever = vectordb.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 5}
                        )
        logger.info("Retriever created.")
        
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        logger.info("Retrieval chain created and ready.")
        return retrieval_chain

    def run_llm(self, input_query, user="user"):
        logger.info(f"run_llm called with input_query: {input_query}")
        if input_query:
            try:
                # Load history for context
                history = self.load_history()
                history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
                logger.info("Creating document chain retriever...")
                chain = self.create_document_chain_retriever(extra_context=history_text)
                logger.info("Invoking chain...")
                response = chain.invoke({"input":input_query, "history": history_text})
                logger.info(f"Chain response: {response}")
                self.append_to_history(input_query, str(response['answer']))
                return response
            except Exception as e:
                logger.error(f"Exception occurred: {e}")
                raise
        else:
            logger.warning("No input_query provided.")
            return None