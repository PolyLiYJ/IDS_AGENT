import os
from typing import Optional

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader

from pathlib import Path
from pprint import pprint
# from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

with open('openai_key.txt', 'r') as file:
    # Read the first line from the file
    first_line = file.readline()
    os.environ["OPENAI_API_KEY"] = first_line

CHROMA_DB_DIRECTORY='db-llama'
DOCUMENT_SOURCE_DIRECTORY='/Users/yanjieli/program/IDS_AGENT/documents'
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    anonymized_telemetry=False
)
TARGET_SOURCE_CHUNKS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HIDE_SOURCE_DOCUMENTS=False
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.vectorstores import Qdrant
import asyncio
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA


class MyKnowledgeBase:
    def __init__(self, pdf_source_folder_path: str) -> None:
        self.pdf_source_folder_path = pdf_source_folder_path

    def load_pdfs(self):
        loader = DirectoryLoader(self.pdf_source_folder_path)
        loaded_pdfs = loader.load()
        return loaded_pdfs

    def split_documents(self, loaded_docs, chunk_size = 1000):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_docs = splitter.split_documents(loaded_docs)
        return chunked_docs

    def convert_document_to_embeddings(self, chunked_docs, embedder):

        vector_db = Chroma.from_documents(documents=chunked_docs, embedding=embedder)
        # vector_db = await Qdrant.afrom_documents(chunked_docs, embeddings, "http://localhost:6333")
        # vector_db.persist()
        #Deprecated since version langchain-community==0.1.17: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
        return vector_db

    def return_retriever_from_persistant_vector_db(self, embedder, chroma_db_directory = None, k = 4):
        if chroma_db_directory is None:
            chroma_db_directory = CHROMA_DB_DIRECTORY
            

        vector_db = Chroma(
            collection_name = 'ids',
            persist_directory=chroma_db_directory,
            embedding_function=embedder,
            # client_settings=CHROMA_SETTINGS,
        )
        
        if not os.path.isdir(CHROMA_DB_DIRECTORY):
            loaded_pdfs = self.load_pdfs()
            print("load file")
            chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
            print("split file")
            vector_db.add_documents(chunked_documents)
            print("add file to the vector_db")
        else:
            print("load vector database")

        return vector_db.as_retriever(search_kwargs={"k": k})

    def initiate_document_injection_pipeline(self, embedder, chroma_db_directory, chunk_size = 1000):
        loaded_pdfs = self.load_pdfs()
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs, chunk_size = chunk_size)

        print("=> PDF loading and chunking done.")

        # embedder = OpenAIEmbeddings()
        
        # ollama_model_name = "mistral:7b-instruct-q6_K"
        # ollama_model_name = "llama2:7b-chat-q6_K"
        # embedder = OllamaEmbeddings(model=ollama_model_name)


        vector_db = Chroma(
            collection_name = 'ids',
            persist_directory=chroma_db_directory,
            embedding_function=embedder,
            # client_settings=CHROMA_SETTINGS,
        )
        

        loaded_pdfs = self.load_pdfs()
        print("load file")
        chunked_documents = self.split_documents(loaded_docs=loaded_pdfs)
        print("split file")
        vector_db.add_documents(chunked_documents)
        print("add file to the vector_db")

        return vector_db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

        print("=> Vector DB initialized and created.")
        print("All done")
        return vector_db

def main():
    # kb = MyKnowledgeBase(pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY)
    # db = kb.initiate_document_injection_pipeline()

    # query = "What is the difference between DoS and DDoS attack?"
    # docs = db.similarity_search(query)
    # print(docs[0].page_content)

    # query = "What is Port Scan attack?"
    # docs = db.similarity_search(query)
    # print(docs[0].page_content)
    chroma_db_database = "db"
    kb = MyKnowledgeBase(pdf_source_folder_path="/Users/yanjieli/program/IDS_AGENT/documents/Recon")
    # ollama_model_name = "llama2:7b-chat-q6_K"
    # embedder = OllamaEmbeddings(model=ollama_model_name)
    # chroma_db_database = 'db-llama'
    # retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    
    # # vector_db = kb.initiate_document_injection_pipeline(embedder)
    # # retriever = vector_db.as_retriever()

    # llm = Ollama(model=ollama_model_name) #, temperature=0.1)
    
    embedder = OpenAIEmbeddings()
    # vector_db = kb.initiate_document_injection_pipeline(embedder, chroma_db_database)
    retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever = retriever)

    print(qa.invoke("what features push the prediction to “recon”?"))
    # print(qa.invoke("The difference between Recon-PortScan and DoS-SYN_Flood"))

if __name__ == "__main__":
    # main()

    chroma_db_database = "db-long-memory"
    # loader = JSONLoader(jq_schema='.action',
    # file_path='/Users/yanjieli/program/IDS_AGENT/longmemory/success_and_fail_example_test_set.json',
    # text_content=False)

    # data = loader.load()
    # vector_db = Chroma(
    #         collection_name = 'longmemory',
    #         persist_directory=chroma_db_directory,
    #         embedding_function=embedder,
    #         # client_settings=CHROMA_SETTINGS,
    #     )
        
    # # loaded_pdfs = self.load_pdfs()
    # print("load file")
    #chunked_documents = self.split_documents(loaded_docs=data)
    # print("split file")
    # vector_db.add_documents(chunked_documents)
    # print("add file to the vector_db")

    # retriever = vector_db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})
    # # ollama_model_name = "llama2:7b-chat-q6_K"
    # # embedder = OllamaEmbeddings(model=ollama_model_name)
    # # chroma_db_database = 'db-llama'
    # # retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    
    # # # vector_db = kb.initiate_document_injection_pipeline(embedder)
    # # # retriever = vector_db.as_retriever()

    # # llm = Ollama(model=ollama_model_name) #, temperature=0.1)
    kb = MyKnowledgeBase("/Users/yanjieli/program/IDS_AGENT/longmemory/text")
    
    embedder = OpenAIEmbeddings()
    #vector_db = kb.initiate_document_injection_pipeline(embedder, chroma_db_database)
    retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever = retriever)
    question =  """retrieve some successful instance of knowledge_retriver tool, which in this format
    Action: knowledge_retriever  
    Action Input: {"query": "..."}  
    Observation: ... """
    print(qa.invoke(question))
