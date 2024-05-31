from dotenv import load_dotenv

load_dotenv()

import os, sys, time, json, datetime

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
            "model_list": os.getenv("MODEL_LIST", ["gpt-4o", "gpt-4-turbo-2024-04-09"]),
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
    ## create an assistant
    def assistant_create(
        self,
        name="My Assistant",
        instructions="You are a personal assistant",
        model=None,
        tools=[],
    ):
        if not model:
            model = self.default_model
        elif model not in self.model_list:
            self.logger.warning(
                f"Model {model} is not in the model list. Using default model {self.default_model} instead."
            )
            model = self.default_model
        assistant = self.client.beta.assistants.create(
            name=name, instructions=instructions, model=model, tools=tools
        )
        return assistant

    ## create a thread
    def assistant_thread_create(self):
        thread = self.client.beta.threads.create()
        return thread

    ## submit a message to an assistant's thread and get the run back
    def assistant_submit_message_to_thread_and_run(self, assistant, thread, message):
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id, content=message, role="user"
        )
        return self.client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant.id
        )

    ## wait for a run of a thread
    # returns the thread and run when the run.status is no longer `queued` or `in_progress`
    # (The status of the run can be either `queued`, `in_progress`,
    # `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`,
    # `incomplete`, or `expired`.)
    def assistant_wait_on_thread_run(self, thread, run, wait_time=3):
        while run.status != "completed":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )
            # self.logger.info(
            #     f"Retrieved run status: run.id: {run.id}, run.status: {run.status}"
            # )
            if run.status == "queued" or run.status == "in_progress":
                time.sleep(wait_time)
            elif run.status != "completed":
                self.logger.warning(
                    f"Run cannot be completed. Run status: {run.status}, last_error: {run.last_error}, incomplete_details: {run.incomplete_details}"
                )
                break
        return thread, run

    ## retrieve all messages of a thread
    def assistant_retrieve_messages_by_thread(self, thread, order="asc"):
        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id, order=order
        )
        return messages

    ## assistant using tools

    ### assistant code interpreter

    ### assistant function calling

    # vector stores


######### test ##########
# test functions
def chat_simple(a: AOAI):
    print("=== simple chat ===")
    response = a.chat(
        query="Hello", system_prompt="You are a very funny and friendly assistant."
    )
    print(response.choices[0].message.content)


def chat_image(a: AOAI, image_path):
    print("=== simple chat with image ===")
    response = a.chat(
        query="Hello, what does this picture say about bears?",
        image=image_path,
        system_prompt="You are a very funny and helpful assistant that can describe pictures.",
        model="gpt-4o",  # same azure endpoint with default model
    )
    print(response.choices[0].message.content)


def assistant_test(a: AOAI):
    assistant = a.assistant_create(
        name="My Assistant",
        instructions="You are a math tutor",
        model="gpt-4o",
    )
    thread = a.assistant_thread_create()
    run = a.assistant_submit_message_to_thread_and_run(
        assistant=assistant, thread=thread, message="What is 2^2?"
    )
    thread, run = a.assistant_wait_on_thread_run(thread, run)
    messages = a.assistant_retrieve_messages_by_thread(thread)
    print("==== Messages in thread ====")
    for m in messages:
        print(f"{m.created_at} | {m.role}: {m.content[0].text.value}")


def assistant_code_interpreter_test(a: AOAI):
    assistant = a.assistant_create(
        name="My Assistant",
        instructions="You are a code interpreter",
        model="gpt-4-turbo-2024-04-09",
        tools=[
            {
                "type": "code_interpreter",
            }
        ],
    )
    thread = a.assistant_thread_create()
    run = a.assistant_submit_message_to_thread_and_run(
        assistant=assistant,
        thread=thread,
        message="Generate the first 20 fibobonaci numbers with code.",
    )
    thread, run = a.assistant_wait_on_thread_run(thread, run)
    messages = a.assistant_retrieve_messages_by_thread(thread)
    for m in messages:
        print(f"{datetime.datetime.fromtimestamp(m.created_at)} | {m.role}: {m.content[0].text.value}")


if __name__ == "__main__":
    aoai = AOAI()

    # chat simple
    # chat_simple(aoai)

    # chat with image
    # chat_image(aoai, "./images/bear.jpg")

    # assistant simple
    # assistant_test(aoai)

    # assistant code interpreter
    assistant_code_interpreter_test(aoai)
