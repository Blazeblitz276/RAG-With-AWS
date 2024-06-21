import boto3  # type: ignore
import streamlit as st  # type: ignore
import os 
import uuid

## S3_client
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

S3_client = boto3.client("s3",region_name= AWS_REGION)

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings # type: ignore
from langchain_community.llms.bedrock import Bedrock # type: ignore

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
bedrock_client = boto3.client(service_name="bedrock-runtime",region_name= AWS_REGION )
bedrock_embedding  = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock_client)



## Creating the vector store
def create_vector_store(request_id, documents):
    vector_store_faiss = FAISS.from_documents( documents, bedrock_embedding)
    FileName= f"{request_id}.bin"
    FolderPath = "/tmp/"
    vector_store_faiss.save_local(index_name= FileName,folder_path= FolderPath, )

    ## Upload the file to S3
    S3_client.upload_file(FolderPath +'/' + FileName + '.faiss', Bucket=  BUCKET_NAME, Key = 'my_faiss.faiss')
    S3_client.upload_file(FolderPath +'/' + FileName + '.pkl', Bucket=  BUCKET_NAME, Key = 'my_faiss.pkl')

    return True

## Folder path for the files
folder_path = "/tmp/"

def load_index():
    S3_client.download_file(Bucket = BUCKET_NAME,Key = "my_faiss.faiss", Filename =f"{folder_path}my_faiss.faiss" )
    S3_client.download_file(Bucket = BUCKET_NAME,Key = "my_faiss.pkl", Filename =f"{folder_path}my_faiss.pkl" )

def get_llm():
    llm = Bedrock(model_id = "meta.llama3-70b-instruct-v1:0", client = bedrock_client,
                  model_kwargs = {"max_gen_len": 512})
    return llm

## Response query 
## prompt and chain
from langchain.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore

def get_response(llm,vectorstore, question):
    ## create prompt / template
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']


def main():
    st.header("This is the Client side for pdf chat demo using Bedrock Titan embeddings")
    
    ## We have to now load th eindex files from S3
    load_index()
    dir_list = os.listdir(folder_path)
    st.write(f"The files in the {folder_path} are ")
    st.write(dir_list)

    ## creating index for the faiss
    faiss_index = FAISS.load_local(
                                    index_name= "my_faiss",
                                    folder_path= folder_path,
                                    embeddings = bedrock_embedding,
                                    allow_dangerous_deserialization = True)
    st.write("The index is loaded successfully")

    questions = st.text_input("Please ask your question") 
    if st.button("Ask a Question"):
        with st.spinner("Querrying......"):
            llm = get_llm()

            #get the question from the user and query the index
            st.write("The question was ", questions)
            st.write("The response is ", get_response(llm,faiss_index, questions))
            st.success("Queried successfully")


    


if __name__ == "__main__":
    main()