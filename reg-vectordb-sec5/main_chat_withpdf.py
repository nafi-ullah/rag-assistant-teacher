# Import necessary libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models import OpenAI

# Step 1: Specify the path to the PDF file
pdf_path = "/Users/edenmarco/Desktop/tmp/react.pdf"

# Step 2: Load the PDF content
loader = PyPDFLoader(file_path=pdf_path)  # Initialize the PDF loader with the file path
documents = loader.load()  # Load the content of the PDF into a variable

# Step 3: Split the loaded text into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,        # Each chunk will have a maximum of 1000 characters
    chunk_overlap=30,       # There will be a 30-character overlap between chunks
    separator="\n"          # Split text at newline characters
)
docs = text_splitter.split_documents(documents=documents)  # Split the text into chunks

# Step 4: Generate embeddings for the text chunks
embeddings = OpenAIEmbeddings()  # Use OpenAI's embedding model to encode the text into vectors

# Step 5: Store the embeddings in a FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)  # Create a FAISS vector store from the documents and embeddings
vectorstore.save_local("faiss_index_react")  # Save the FAISS vector store locally for future use

# Step 6: Load the FAISS vector store back from the saved file
new_vectorstore = FAISS.load_local(
    "faiss_index_react",                     # Path to the saved FAISS index
    embeddings,                              # Use the same embedding model for loading
    allow_dangerous_deserialization=True     # Allow deserialization for testing (use cautiously in production)
)

# Step 7: Download a prebuilt retrieval QA chat prompt from LangChain's hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Step 8: Create a chain to combine retrieved documents into a cohesive response
combine_docs_chain = create_stuff_documents_chain(
    OpenAI(),                    # Use OpenAI's language model to generate responses
    retrieval_qa_chat_prompt     # Use the downloaded prompt template
)

# Step 9: Create the retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever=new_vectorstore.as_retriever(),  # Convert the vector store into a retriever
    combine_docs_chain=combine_docs_chain     # Combine the retrieved documents into a response
)

# Step 10: Use the retrieval chain to process a query and generate an answer
res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})

# Step 11: Print the final answer generated by the retrieval chain
print(res["answer"])
