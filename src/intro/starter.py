import boto3
import pprint

bedrock = boto3.client(service_name="bedrock", region_name="eu-west-1")

pp = pprint.PrettyPrinter(depth=4)


def list_foundation_models():
    models = bedrock.list_foundation_models()
    print("Available models:")
    for model in models["modelSummaries"]:
        print(f"- {model['modelId']}")
    return models


def get_foundation_model(modelIdentifier):
    try:
        model = bedrock.get_foundation_model(modelIdentifier=modelIdentifier)
        pp.pprint(model)
    except Exception as e:
        print(f"Error getting model {modelIdentifier}: {e}")


print("Listing available models...")
list_foundation_models()

print("\nTrying to get a model...")
# get_foundation_model('anthropic.claude-3-sonnet-20240229-v1:0')
get_foundation_model("meta.llama2-13b-chat-v1")
