import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import time

class IngestData():
    def __init__(self):
        self.DB_PATH = ".\src\core\\vectorDB"
        print(f"[IngestData] Initialized with DB_PATH: {self.DB_PATH}")

    def load_documents(self, file_paths):
        print(f"[IngestData] Loading documents from: {file_paths}")
        all_documents = []
        for file_path in file_paths:
            print(f"[IngestData] Loading file: {file_path}")
            document_loader = PyPDFLoader(file_path=file_path)
            loaded = document_loader.load()
            print(f"[IngestData] Loaded {len(loaded)} documents from {file_path}")
            all_documents.extend(loaded)
        print(f"[IngestData] Total documents loaded: {len(all_documents)}")
        return all_documents


    def split_documents(self, documents):
        print(f"[IngestData] Splitting {len(documents)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[IngestData] Split into {len(chunks)} chunks.")
        return chunks


    def ingest_documents(self, chunks):
        print(f"[IngestData] Ingesting {len(chunks)} chunks into vector DB...")
        vectordb = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=self.DB_PATH)
        if len(chunks):
            print(f"[IngestData] Adding {len(chunks)} new documents in the DB...")
            vectordb.persist()
            print("[IngestData] Vector DB persisted.")
        else:
            print("[IngestData] No documents to add.")


    def clear_database(self):
        if os.path.exists(self.DB_PATH):
            print("[IngestData] Clearing database...")
            shutil.rmtree(self.DB_PATH)
            print("[IngestData] Database cleared.")
        else:
            print("[IngestData] No database to clear.")
    
    def ingest_data(self, files):
        print(f"[IngestData] Starting ingestion for files: {files}")
        start_time = time.process_time()
        self.clear_database()
        documents = self.load_documents(files)
        chunks = self.split_documents(documents)
        self.ingest_documents(chunks)
        print(f"[IngestData] Process completed in {time.process_time() - start_time}s")
