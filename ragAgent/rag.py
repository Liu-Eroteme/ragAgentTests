from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import chromadb


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # AB test 2 versions of modelfile
        self.model = ChatOllama(model="hermes2pro") #with defined system prompt template
        # self.model = ChatOllama(model="hermes2proV2") #without defined system prompt template
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        
        self.prompt = PromptTemplate.from_template(
            """
            <|im_start|>system
            You are an assistant for answering questions. Use the following context elements to answer the question.
            If you don't know the answer, simply say that you don't know. Use a maximum of three sentences
            and be concise in your response.<|im_end|>
            <|im_start|>user
            Question: {question}
            Context: {context}<|im_end|>
            <|im_start|>assistant
            """
        )
        
        model_name = "intfloat/multilingual-e5-large"
        model_kwargs = {'device': 'gpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def ingest(self):
        folderPath = "./ragData"
        documents =[]
        for file in os.listdir(folderPath):
                if file.endswith(".pdf"):
                        pdfPath = os.path.join(folderPath, file)
                        loader = PyPDFLoader(pdfPath)
                        documents.extend(loader.load())
        chunkedDocs = self.text_splitter.split_documents(documents)
        chunkedDocs = self.text_splitter.split_documents(chunkedDocs)
        vector_store = Chroma.from_documents(documents=chunkedDocs, embedding=self.hf)
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None