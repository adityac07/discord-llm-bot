import re
import os
import json
import openai
import aiohttp
import google.generativeai as genai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
# TODO: MAKE SURE TO COMMENT THIS OUT AFTER DONE TESTING
load_dotenv()

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Configure APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")


class LLMHandler:
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove HTML-like tags from the response. Need for deepseek R1 model if its used in ollama"""
        cleaned_text: str = re.sub(r"<[^>]+>", "", text)
        return cleaned_text.strip()

    @staticmethod
    async def get_ollama_response(prompt: str, context: list = None, model: str = OLLAMA_MODEL) -> str:
        """Get response from Ollama API"""
        try:
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_API_URL}/api/chat",
                    json={"model": model, "messages": messages},
                ) as response:
                    if response.status == 200:
                        full_response: str = ""
                        async for line in response.content:
                            line: str = line.decode("utf-8").strip()
                            if line:
                                try:
                                    json_response: dict = json.loads(line)
                                    if "message" in json_response and "content" in json_response["message"]:
                                        full_response += json_response["message"]["content"]
                                except json.JSONDecodeError:
                                    continue

                        cleaned_response: str = LLMHandler.clean_response(full_response)
                        return (
                            cleaned_response
                            if cleaned_response
                            else "Error: No valid response received"
                        )
                    else:
                        logging.error(
                            f"API returned status code {response.status}"
                        )
                        return None
        except Exception as e:
            logging.error(f"{str(e)}")
            return

    @staticmethod
    async def get_openai_response(prompt: str, context: list = None, model: str = OPENAI_MODEL) -> str:
        """Get response from OpenAI or OpenAI-compatible API"""
        try:
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            client = AsyncOpenAI(api_key=openai.api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"{str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    async def get_gemini_response(prompt: str, context: list = None, model: str = GEMINI_MODEL) -> str:
        """Get response from Google Gemini API"""
        try:
            # Gemini doesn't support chat history in the same way, so we'll concatenate context
            full_prompt = ""
            if context:
                for message in context:
                    role = "User" if message["role"] == "user" else "Assistant"
                    full_prompt += f"{role}: {message['content']}\n\n"
            full_prompt += f"User: {prompt}"

            model = genai.GenerativeModel(model)
            response = await model.generate_content_async(full_prompt)

            # Check if response has parts
            if not response.parts:
                logging.error("Error: No response generated")
                return

            # Concatenate all parts of the response
            full_response = ""
            for part in response.parts:
                if hasattr(part, "text"):
                    full_response += part.text

            # If we got no text, check for other response formats
            if not full_response and hasattr(response, "candidates"):
                for candidate in response.candidates:
                    if hasattr(candidate, "content"):
                        full_response += candidate.content.parts[0].text

            return (
                full_response
                if full_response
                else "Error: Could not extract response text"
            )

        except Exception as e:
            logging.error(f"Error in Gemini response: {str(e)}")
            return

    @staticmethod
    async def get_deepseek_response(prompt: str, context: list = None) -> str:
        """Get response from DeepSeek API"""
        try:
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            client = AsyncOpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            if hasattr(response.choices[0].message, "content"):
                return response.choices[0].message.content
            return "Error: No response content received"
        except Exception as e:
            error_message: str = f"Error in DeepSeek response: {str(e)}"
            logging.error(error_message)
            return error_message
