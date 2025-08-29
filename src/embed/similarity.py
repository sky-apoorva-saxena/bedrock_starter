import boto3
import json


client = boto3.client(service_name="bedrock-runtime", region_name="eu-west-1")

fact = "the capital of france is paris"

animal = "cat"

response = client.invoke_model(
    body=json.dumps({"inputText": fact}),
    modelId="amazon.titan-embed-text-v2:0",
    accept="application/json",
    contentType="application/json",
)


response_body = json.loads(response.get("body").read())
print(response_body.get("embedding"))
