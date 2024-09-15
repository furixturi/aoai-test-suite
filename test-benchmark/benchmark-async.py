import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import argparse
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    filename='aoai_benchmark.log',
    filemode='w',  # Overwrite the log file each time the script runs
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Load environment variables from .env file
load_dotenv()
API_ENDPOINT_ENV = os.environ["GPT_4o_SWC_20240913_ENDPOINT"]
DEPLOYMENT = os.environ["GPT_4o_SWC_20240913_PTU_DEPLOYMENT"]
# DEPLOYMENT = os.environ["GPT_4o_SWC_20240913_GLOBAL_PAYGO_DEPLOYMENT"]
API_KEY = os.environ["GPT_4o_SWC_20240913_KEY"]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="AOAI Request Benchmark Script")
# parser.add_argument('--prompt', type=str, required=True, help='Prompt to send to the model')
parser.add_argument('--tokens', type=int, required=True, help='Number of tokens to generate')
parser.add_argument('--requests', type=int, default=100, help='Total number of requests to send')
parser.add_argument('--rpm', type=int, default=20, help='Requests per minute')
args = parser.parse_args()
tokens = args.tokens

# Azure OpenAI API settings
API_VERSION = "2024-06-01"  # Ensure this is the correct API version
API_ENDPOINT = f"{API_ENDPOINT_ENV}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}&generated_tokens={tokens}"
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}



async def make_request(session, i, prompt, tokens):
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant trained to generate exactly the number of tokens requested. Always generate content with deep detail and explanation to ensure the required token count is reached. Don't output anything else."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": tokens,
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    request_time = datetime.now(timezone.utc)
    request_time_str = request_time.isoformat()

    try:
        async with session.post(API_ENDPOINT, headers=HEADERS, json=data) as response:
            if response.status == 200:
                result = await response.json()
                
                response_time = datetime.now(timezone.utc)
                response_time_str = datetime.now(timezone.utc).isoformat()
                
                duration = (response_time - request_time).total_seconds()
                
                content = result["choices"][0]["message"]["content"]
                completion_tokens = result["usage"]["completion_tokens"]
                
                logging.info(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} |  Duration: {duration:.3f}s | Completion tokens: {completion_tokens} | Response: {content}" 
                )
                
                print(f'Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Completion tokens: {completion_tokens}')
                return duration
            else:
                error_text = await response.text()
                
                response_time = datetime.now(timezone.utc)
                response_time_str = datetime.now(timezone.utc).isoformat()
                
                duration = (response_time - request_time).total_seconds()
                
                logging.error(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Error: {error_text}"
                )
                print(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Error: {error_text}"
                )
                return None
                
    except Exception as e:
        error_time = datetime.now(timezone.utc)
        error_time_str = error_time.isoformat()
        duration = (error_time - request_time).total_seconds() 
        
        logging.exception(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Duration: {duration:.3f} | Exception: {e}"
        )
        print(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Duration: {duration:.3f} | Exception: {e}"
        )
        return None
        
async def scheduler(session, total_requests, prompt, tokens, delay_between_calls):
    tasks = []
    for i in range(total_requests):
        request_start_time = i * delay_between_calls
        task = asyncio.create_task(schedule_request(session, i+1, prompt, tokens, request_start_time))
        tasks.append(task)
    durations = await asyncio.gather(*tasks) # execute all tasks concurrently
    return durations

async def schedule_request(session, request_number, prompt, tokens, delay):
    await asyncio.sleep(delay)
    duration = await make_request(session, request_number, prompt, tokens)
    return duration

async def main():
    delay_between_calls = 60 / args.rpm  # seconds

    total_requests = args.requests
    tokens = args.tokens
    # prompt = f"Generate at least {tokens} tokens in Japanese about the history, structure, technologies, and future of banking. Be as detailed as possible in your explanations to ensure the content reaches the token limit. Don’t stop until the requested token count is reached."
    
    prompt = f"Generate {tokens} tokens in Japanese about banks, of which {tokens//4} about Japan bank history, {tokens//4} about different businesses that a bank in Japan runs, {tokens//4}  about operations and processes, {tokens//4}  about new technologies. Don’t stop until the requested token count is reached."
    

    async with aiohttp.ClientSession() as session:
        durations = await scheduler(session, total_requests, prompt, tokens, delay_between_calls)
        success_durations = [d for d in durations if d is not None]
        if success_durations:
            average_duration = sum(success_durations) / len(success_durations)
            print(f"Successful requests made: {len(success_durations)} | Average duration: {average_duration:.3f}s | RPM: {args.rpm}")
            logging.info(f"Average duration: {average_duration:.3f}s - RPM: {args.rpm}")
        else:
            print("No successful requests were made.")
            logging.info("No successful requests were made.")

if __name__ == '__main__':
    asyncio.run(main())
