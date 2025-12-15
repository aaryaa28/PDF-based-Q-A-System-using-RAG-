from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import boto3

def get_embedding_function():
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'  # replace with your AWS region
    )
    
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v1"
    )
    
    return embeddings
