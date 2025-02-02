# Import necessary libraries and modules
import os  # Provides functions to interact with the operating system
from dotenv import load_dotenv  # Loads environment variables from a .env file
from langchain_core.prompts import PromptTemplate  # For creating prompt templates
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI embeddings and chat model
from langchain_pinecone import PineconeVectorStore  # Interface for storing and retrieving vectors in Pinecone

# LangChain's hub for downloading reusable prompts and chains
from langchain import hub

# Functions to combine and retrieve documents
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load environment variables from a .env file
load_dotenv()

# Main function to execute the code
if __name__ == "__main__":
    print(" Retrieving...")  # Indicate that the retrieval process is starting

    # Create an instance of OpenAIEmbeddings for generating embeddings from text
    embeddings = OpenAIEmbeddings()

    # Create an instance of ChatOpenAI for interacting with OpenAI's chat model
    llm = ChatOpenAI()

    # Define the user query
    query = "what is Pinecone in machine learning?"

    # Create a chain that links the query directly to the LLM
    # (This step directly uses the query as the prompt for the model)
    chain = PromptTemplate.from_template(template=query) | llm
    # Uncomment the following lines to run the above chain
    # result = chain.invoke(input={})
    # print(result.content)

    # Create a Pinecone vector store to store and retrieve embeddings
    # The vector store is initialized with an index name (from environment variables) and embeddings
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],  # Retrieve the Pinecone index name from the environment
        embedding=embeddings  # Use the OpenAIEmbeddings instance for encoding text
    )

    # Download a prebuilt retrieval-augmented QA prompt template from LangChain's hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    '''
    Answer any use questions based solely on the context below:
        <context>
            {context}
        </context>
    '''
    # Create a chain to combine retrieved documents into a cohesive response
    combine_docs_chain = create_stuff_documents_chain(
        llm,  # The language model to generate responses
        retrieval_qa_chat_prompt  # The prompt template downloaded from the hub
    )

    # Create a retrieval chain that links a retriever (from the vector store) to the document combining chain
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),  # Convert the vector store into a retriever
        combine_docs_chain=combine_docs_chain  # The chain for combining retrieved documents
    )

    # Invoke the retrieval chain with the user's query as input
    result = retrival_chain.invoke(input={"input": query})

    # Print the final result
    print(result)
