import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# the document loaders are actually classes implementations of how to load and process data in order to make it digestible by the large language model.
# https://python.langchain.com/docs/concepts/#document-loaders

from langchain_text_splitters import CharacterTextSplitter
# Now text splitters are here to help us with long pieces of text.
# Those texts have tons of tokens inside them and if we'll send them directly to the LLM, then our request will probably fail because it surpassed the token limitation the model enforces.
# So for example, in the GPT-3.5,we have 4K tokens limitation.So to conclude, the text splitter allows us to take text which is large and to split it into chunk. Now to be honest, text splitters
# have a lot of logic in there because there are a lot of splitting strategies and there are a lot of smart ways to do it with calculating the appropriate chunk size. Now the chunk size is not trivial because it may change according to what we want to accomplish.
# https://python.langchain.com/docs/how_to/character_text_splitter/

from langchain_openai import OpenAIEmbeddings
# https://python.langchain.com/docs/how_to/embed_text/
from langchain_pinecone import PineconeVectorStore
# https://python.langchain.com/docs/integrations/vectorstores/pinecone/
load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

