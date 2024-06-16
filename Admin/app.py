import boto3
import streamlit as st
import os 
import uuid

## S3_client

S3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
 

 ## PDF loader 


def main():
    st.write("This is the Admin page for pdf demo")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        saved_file_name = f"{request_id}.pdf" 
        with open(saved_file_name, "wb") as w:
            w.write(uploaded_file.getvalue())

        ## Load the PDF
        pdf_loader = PyPDFLoader(saved_file_name)
        pages = pdf_loader.load_and_split()






if __name__ == "__main__":
    main()