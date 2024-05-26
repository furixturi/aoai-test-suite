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
            "model_list": os.getenv("MODEL_LIST", []),
        }
        self.configs = utilities.merge_configs(
            default_configs=default_configs,
            envs=os.environ,
            configs=configs,
            required_keys=["api_key", "api_version", "azure_endpoint"],
        )

        self.default_model = self.configs["default_model"]
        self.model_list = (
            self.configs.get("model_list")
            if len(self.configs.get("model_list")) > 0
            else [self.default_model]
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

    def get_model_list(self):
        return self.model_list

    def set_model_list(self, model_list):
        if model_list is not None and isinstance(model_list, list):
            self.logger.info(f"Setting model list to {model_list}.")
            self.model_list = model_list
        else:
            self.logger.warning(
                f"Cannot set model list to {model_list}, which is not a list."
            )

    def add_model(self, model):
        self.logger.info(f"Adding model {model} to the model list.")
        self.model_list.append(model)

    def remove_model(self, model):
        if model in self.model_list:
            self.logger.info(f"Removing model {model} from the model list.")
            self.model_list.remove(model)
        else:
            self.logger.info(
                f"Cannot remove model {model}, which is not in the model list."
            )

    def get_default_model(self):
        return self.default_model

    def set_default_model(self, model):
        if model in self.model_list:
            self.logger.info(f"Changing default model to {model}.")
            self.default_model = model
        else:
            self.logger.info(
                f"Cannot set default model to {model}, which is not in the model list."
            )

    def chat(
        self,
        query,
        image=None,
        model=None,
        system_prompt="You're a helpful assistant",
        chat_history=[],
        stream=False,
        # stream_options={},   # not supported as in API version 2024-05-01-preview. Github issue: https://github.com/Azure/azure-rest-api-specs/issues/29157
        response_format="text",
    ):
        # sanity check and fallback mechanism
        if not model:
            model = self.default_model
        elif model not in self.model_list:
            self.logger.warning(
                f"Model {model} is not in the model list. Using default model {self.default_model} instead."
            )
            model = self.default_model
        # prepare messages
        messages = []
        # prepare system prompt
        messages.append({"role": "system", "content": system_prompt})
        # add history
        messages.extend(chat_history)
        # add user query
        messages.append(self._create_multimodal_prompt_object(query, image))
        # get chat response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            response_format={"type": response_format},
            # stream_options=stream_options
        )
        return response

    def chat_with_tools(
        self,
        query,
        tools,
        tool_choice="auto",
        model=None,
        system_prompt="You're a helpful assistant",
        chat_history=[],
    ):
        # sanity check and fallback mechanism
        if not model:
            model = self.default_model
        elif model not in self.model_list:
            self.logger.warning(
                f"Model {model} is not in the model list. Using default model {self.default_model} instead."
            )
            model = self.default_model
        # prepare messages
        messages = []
        # prepare system prompt
        messages.append({"role": "system", "content": system_prompt})
        # add history
        messages.extend(chat_history)
        # add user query
        messages.append(self._create_multimodal_prompt_object(query))
        # get chat response
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice
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
        model="gpt-4o",  # same azure endpoint with default model
    )
    print(response.choices[0].message.content)
