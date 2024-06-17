import boto3  # type: ignore
import streamlit as st  # type: ignore
import os 
import uuid

## S3_client

S3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings # type: ignore

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter# type: ignore
 

 ## PDF loader 
from langchain_community.document_loaders import PyPDFLoader# type: ignore

## Unique ID generation 
def get_unique_id():
    return str(uuid.uuid4())

## splitting the pages and texts 
def split_text(pages, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages)
    return docs

##import FAISS
from langchain_community.vectorstores import FAISS # type: ignore
bedrock_client = boto3.client("service_name=bedrock-runtime")
bedrock_embedding  = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock_client)



## Creating the vector store
def create_vector_store(request_id, documents):
    vector_store_faiss = FAISS.from_documents(request_id, documents, bedrock_embedding)
    FileName= f"{request_id}.bin"
    FolderPath = "/tmp/"
    vector_store_faiss.save_local(index_name= FileName,folder_path= FolderPath, )

    ## Upload the file to S3
    S3_client.upload_file(FolderPath +'/' + FileName + '.faiss', Bucket=  BUCKET_NAME, Key = 'my_faiss.faiss')
    S3_client.upload_file(FolderPath +'/' + FileName + '.pkl', Bucket=  BUCKET_NAME, Key = 'my_faiss.pkl')

    return True


def main():
    st.write("This is the Admin page for pdf demo")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        request_id = get_unique_id
        st.write("File uploaded, Name:", uploaded_file.name, "Size:", uploaded_file.size)
        st.write("Request ID:", {request_id}) 
        saved_file_name = f"{request_id}.pdf" 
        with open(saved_file_name, "wb") as w:
            w.write(uploaded_file.getvalue())

        ## Load the PDF
        pdf_loader = PyPDFLoader(saved_file_name)
        pages = pdf_loader.load_and_split()

        st.write("Number of pages in the PDF:", len(pages)) 

        ## Split the text     
        splitted_doc = split_text(pages, 1000, 200)
        st.write(f"Splitted doc length: {len(splitted_text)} ")
        st.write("===========================================")
        st.write(splitted_doc[0])
        st.write("===========================================")
        st.write(splitted_doc[1])

        st.write("Creating the vector store")
        result = create_vector_store(request_id, splitted_doc)
        if result:
            st.write("Vector store created successfully")
        else:
            st.write("Error in creating the vector store")
        



if __name__ == "__main__":
    main()