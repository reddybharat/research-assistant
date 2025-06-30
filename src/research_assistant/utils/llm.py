import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

from dotenv import load_dotenv
load_dotenv()

Ollama.model_rebuild()

class LLM():
    def __init__(self):
        print("[LLM] Initializing LLM class...")
        self.DB_PATH = '.\src\core\\vectorDB'
        print(f"[LLM] Initialization complete. DB_PATH: {self.DB_PATH}")

    def create_document_chain_retriever(self):
        print("[LLM] Creating document chain retriever...")
        vectordb = Chroma(persist_directory=self.DB_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        print("[LLM] VectorDB created.")

        user_prompt = ChatPromptTemplate.from_template(
        """
        Elaborate and answer the following question based only on the provided context.
        Think step by step before providing an answer.
        Please do not go out of context to answer the question, if the answer is not present in the given context then you dont have to answer the question.
        <context> {context} </context>
        Question : {input}
        """
        )
        print("[LLM] User prompt template created.")

        # model = ChatOpenAI(model=self.model, api_key=self.api_key)
        model_ollama = Ollama(model='llama3')
        print("[LLM] Ollama model created.")
        doc_chain = create_stuff_documents_chain(model_ollama, user_prompt)
        print("[LLM] Document chain created.")

        retriever = vectordb.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 5}
                        )
        print("[LLM] Retriever created.")
        
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
        print("[LLM] Retrieval chain created and ready.")
        return retrieval_chain

    def run_llm(self, input_query):
        print(f"[LLM] run_llm called with input_query: {input_query}")
        if input_query:
            try:
                print("[LLM] Creating document chain retriever...")
                chain = self.create_document_chain_retriever()
                print("[LLM] Invoking chain...")
                response = chain.invoke({"input":input_query})
                print(f"[LLM] Chain response: {response}")
                return response
            except Exception as e:
                print(f"[LLM] Exception occurred: {e}")
                raise
        else:
            print("[LLM] No input_query provided.")
            return None