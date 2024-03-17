from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
import os
import chromadb


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    # ----- helper funcs -----
    def _load_and_create_search_tool(self, subfolder, tool_name, description):
        subfolderPath = os.path.join("./ragData", subfolder)
        documents = []

        for file in os.listdir(subfolderPath):
            if file.endswith(".pdf"):
                pdfPath = os.path.join(subfolderPath, file)
                loader = PyPDFLoader(pdfPath)
                documents.extend(loader.load())

        chunkedDocs = self.text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=chunkedDocs, embedding=self.hf)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )

        return create_retriever_tool(
            retriever=retriever,
            name=tool_name,
            description=description,
        )

    # ----- /helper funcs -----

    def __init__(self):
        # FIXME BODGE ALERT: changes made in package:
        # site-packages/langchain_community/llms/ollama.py
        # added 2 lines:
        # below line 188: "stop = None"
        # below line 210: "params["options"]["stop"] = stop"
        # These lines ensure that the final 'stop' tokens, determined after considering
        # both default and input values, are explicitly set in the API parameters.
        # It resolves issues where 'stop' tokens were either incorrectly merged or
        # ignored due to the complex logic of merging 'kwargs' and default parameters.
        # This change is crucial for ensuring that Ollama API calls correctly recognize
        # and utilize the intended 'stop' tokens

        # AB test 2 versions of modelfile
        # self.model = ChatOllama(model="hermes2pro") #with defined system prompt template
        self.model = Ollama(
            model="hermes2proV2", verbose=True, stop=["<|im_end|>", "Observation:"]
        )  # without defined system prompt template

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=128
        )

        self.prompt = PromptTemplate.from_template(
            """
            <|im_start|>system
            You are a knowledgeable assistant equipped with various tools to help answer questions.
            When faced with a question, think carefully about the best way to approach it.
            Utilize the tools available to you to gather information and form your own conclusions.
            Your tools are:
            
            {tools}
            
            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!<|im_end|>

            <|im_start|>user
                Question: {input}<|im_end|>

            <|im_start|>assistant
                {agent_scratchpad}
            """
        )

        model_name = "intfloat/multilingual-e5-large"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": False}
        self.hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress=True,
        )

    def ingest(self):
        common_law_tool = self._load_and_create_search_tool(
            "commonLaw",
            "search_common_law_knowledge_base",
            "Searches and returns excerpts from your common law knowledge base.",
        )
        trust_law_tool = self._load_and_create_search_tool(
            "trustLaw",
            "search_trust_law_knowledge_base",
            "Searches and returns excerpts from your trust law knowledge base.",
        )

        self.tools = [common_law_tool, trust_law_tool]

        self.react_agent = create_react_agent(self.model, self.tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def ask(self, query: str):
        if not self.agent_executor:
            return "Please, add a PDF document first."
        formattedQuery = {"input": query}
        result = self.agent_executor.invoke(formattedQuery)
        print("result: " + str(result))
        return result["output"]

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
