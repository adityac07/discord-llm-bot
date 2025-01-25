import os
from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv
from llmhandler import LLMHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Load environment variables
# TODO: MAKE SURE TO COMMENT THIS OUT AFTER DONE TESTING
load_dotenv()

# get env variables
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER")

# Bot setup
intents: Intents = Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=None, intents=intents)


@bot.event
async def on_ready() -> None:
    logging.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message) -> None:
    """Respond to mentions"""
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        if not question:
            await message.reply("Please ask a question after mentioning me.")
            return

        # Use the configured default provider
        if DEFAULT_LLM_PROVIDER == "ollama":
            response = await LLMHandler.get_ollama_response(question)
        elif DEFAULT_LLM_PROVIDER in ["openai", "openai-compatible"]:
            response = await LLMHandler.get_openai_response(question)
        elif DEFAULT_LLM_PROVIDER == "gemini":
            response = await LLMHandler.get_gemini_response(question)
        elif DEFAULT_LLM_PROVIDER == "deepseek":
            response = await LLMHandler.get_deepseek_response(question)
        else:
            response = "Invalid LLM provider configured"

        if not response:
            await message.reply(
                "Sorry, I encountered an error processing your request."
            )
            return

            # Split response into chunks if it's too long
        max_length = 2000
        first_chunk = True
        for i in range(0, len(response), max_length):
            chunk = response[i : i + max_length]
            if first_chunk:
                await message.reply(chunk)
                first_chunk = False
            else:
                await message.channel.send(chunk)


# Run the bot
bot.run(os.getenv("DISCORD_TOKEN"))
