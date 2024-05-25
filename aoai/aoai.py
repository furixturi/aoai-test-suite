from dotenv import load_dotenv

load_dotenv()

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai import AzureOpenAI, AsyncAzureOpenAI

import utilities.utilities as utilities

# from utilities.utilities import merge_configs, setup_logger


class AOAI:
    """AOAI class
    Create an instance for each endpoint
    The model (deployment) is to be specified when calling the methods
    """

    def __init__(self, configs={}) -> None:
        default_configs = {
            "max_tokens": 2000,
            "api_key": os.getenv("DEFAULT_API_KEY"),
            "azure_endpoint": os.getenv("DEFAULT_AZURE_ENDPOINT"),
            "api_version": os.getenv("DEFAULT_API_VERSION"),
            "default_model": os.getenv("DEFAULT_MODEL"),
        }
        self.configs = utilities.merge_configs(
            default_configs=default_configs,
            envs=os.environ,
            configs=configs,
            required_keys=["api_key", "api_version", "azure_endpoint"],
        )
        self.logger = utilities.setup_logger()
        self._createAOAIClients()

    def _createAOAIClients(self):
        api_key = self.configs["api_key"]
        api_version = self.configs["api_version"]
        azure_endpoint = self.configs["azure_endpoint"]
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.client_async = AsyncAzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
        )
        self.default_model = self.configs["default_model"]

    # default
    ## chat
    ### default model is set in the env DEFAULT_MODEL
    def chat(
        self,
        query,
        model=None,
        system_prompt="You're a helpful assistant",
        chat_history=[],
        stream=False,
    ):
        if not model:
            model = self.default_model
        messages = []
        # prepare system prompt
        messages.append({"role": "system", "content": system_prompt})
        # add history
        messages.extend(chat_history)
        # add user query
        messages.append({"role": "user", "content": query})
        # get chat response
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=stream
        )
        return response

    def chat_image_input(
        self,
        query,
        image,
        model=None,
        system_prompt="You're a helpful assistant",
        chat_history=[],
        stream=False,
    ):
        if not model:
            model = self.default_model
        messages = []
        # prepare system prompt
        messages.append({"role": "system", "content": system_prompt})
        # add history
        messages.extend(chat_history)
        # add user query
        messages.append(self._create_multimodal_prompt_object(query, image))
        # get chat response
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=stream
        )
        return response

    def _create_multimodal_prompt_object(self, query, image=None, role="user"):
        prompt_object = {
            "role": role,
            "content": [{"type": "text", "text": query}]
            + (
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{utilities.encode_image(image)}"
                            # "url": image_url
                        },
                    }
                ]
                if image
                else []
            ),
        }

        return prompt_object

    ## audio
    ### transcription (stt)

    ### speech (tts)

    ### translation

    ## image generation

    # function calling

    # assistants

    # vector stores


if __name__ == "__main__":
    aoai = AOAI()
    print("simple chat")
    response = aoai.chat(
        query="Hello", system_prompt="You are a very funny and friendly assistant."
    )
    print(response.choices[0].message.content)
    
    print("simple chat with image")
    response = aoai.chat_image_input(
        query="Hello, what does this picture say about bears?",
        image="./images/bear.jpg",
        system_prompt="You are a very funny and helpful assistant that can describe pictures.",
        model="gpt-4o" # same azure endpoint with default model
    )
    print(response.choices[0].message.content)