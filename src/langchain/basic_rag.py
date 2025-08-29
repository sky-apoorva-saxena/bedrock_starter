from langchain_aws import BedrockLLM as Bedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3


my_data = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Don likes to eat pizza.",
    "Don likes to eat pasta.",
]

question = "What does Don like to eat?"


AWS_REGION = "eu-west-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

# create embeddings

vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

# create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.get_relevant_documents(question)

result_strings = []
for result in results:
    result_strings.append(result.page_content)

# build template
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),
    ]
)

chain = template.pipe(model)

resposnse = chain.invoke({"input": question, "context": result_strings})
print(resposnse)
