import re
import os
import json
import openai
import aiohttp
import google.generativeai as genai
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import logging
import httpx

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
    def __init__(self):
        # Load API configurations
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        # Default system prompts
        self.default_system_prompts = {
            "ollama": "You are a helpful AI assistant. Be concise and clear in your responses. Hide your identity",
            "openai": "You are a helpful AI assistant. Be concise and clear in your responses. Hide your identity",
            "gemini": "You are a helpful AI assistant. Be concise and clear in your responses. Hide your identity",
            "deepseek": "You are a helpful AI assistant. Be concise and clear in your responses. Hide your identity",
        }

        # Initialize custom system prompts (can be modified at runtime)
        self.custom_system_prompts = {}

    def set_system_prompt(self, provider: str, prompt: str) -> None:
        """Set a custom system prompt for a specific provider"""
        self.custom_system_prompts[provider] = prompt

    def get_system_prompt(self, provider: str) -> str:
        """Get the system prompt for a provider (custom if set, otherwise default)"""
        return self.custom_system_prompts.get(
            provider, self.default_system_prompts.get(provider, "")
        )

    @staticmethod
    def prepare_messages(
        system_prompt: str, user_prompt: str, context: list = None
    ) -> list:
        """Prepare messages list with system prompt, context, and user prompt"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def clean_response(text: str) -> str:
        """Remove HTML-like tags from the response. Need for deepseek R1 model if its used in ollama"""
        cleaned_text: str = re.sub(r"<[^>]+>", "", text)
        return cleaned_text.strip()

    @staticmethod
    async def get_ollama_response(
        prompt: str, context: list = None, system_prompt: str = None
    ) -> str:
        """Get response from Ollama API with system prompt support"""
        try:
            messages = LLMHandler.prepare_messages(system_prompt, prompt, context)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{os.getenv('OLLAMA_API_URL')}/api/chat",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "deepseek-r1:14b"),
                        "messages": messages,
                    },
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["message"]["content"]
                    return f"Error: API returned status code {response.status}"
        except Exception as e:
            logging.error(f"Error in Ollama response: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    async def get_openai_response(
        prompt: str, context: list = None, system_prompt: str = None
    ) -> str:
        """Get response from OpenAI API with system prompt support"""
        try:
            messages = LLMHandler.prepare_messages(system_prompt, prompt, context)

            async with httpx.AsyncClient() as session:
                client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    http_client=session
                )
                response = await client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), 
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in OpenAI response: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    async def get_gemini_response(
        prompt: str, context: list = None, system_prompt: str = None
    ) -> str:
        """Get response from Gemini API with system prompt support"""
        try:
            # Combine system prompt, context, and user prompt
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            if context:
                for msg in context:
                    full_prompt += f"{msg['role']}: {msg['content']}\n"
            full_prompt += prompt

            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
            response = await model.generate_content_async(full_prompt)

            if not response.parts:
                return "Error: No response generated"

            full_response = ""
            for part in response.parts:
                if hasattr(part, "text"):
                    full_response += part.text

            return (
                full_response
                if full_response
                else "Error: Could not extract response text"
            )
        except Exception as e:
            logging.error(f"Error in Gemini response: {str(e)}")
            return f"Error: {str(e)}"

    @staticmethod
    async def get_deepseek_response(
        prompt: str, context: list = None, system_prompt: str = None
    ) -> str:
        """Get response from DeepSeek API with system prompt support"""
        try:
            messages = LLMHandler.prepare_messages(system_prompt, prompt, context)

            async with httpx.AsyncClient() as session:
                client = AsyncOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                    http_client=session
                )
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                )
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in DeepSeek response: {str(e)}")
            return f"Error: {str(e)}"
