import os
from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv
from llmhandler import LLMHandler
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Load environment variables
# TODO: MAKE SURE TO COMMENT THIS OUT AFTER DONE TESTING
load_dotenv()

# get env variables
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER")
USER_SYSTEM_PROMPT: str = os.getenv("USER_SYSTEM_PROMPT")
# Bot setup
intents: Intents = Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=commands.when_mentioned_or(""), intents=intents)

# Context storage
user_contexts = defaultdict(list)
MAX_CONTEXT_LENGTH = 5  # Keep last 5 messages as context

# Initialize LLMHandler
llm_handler = LLMHandler()

# Set custom system prompts if needed
llm_handler.set_system_prompt(
    DEFAULT_LLM_PROVIDER,
    USER_SYSTEM_PROMPT,
)


@bot.event
async def on_ready() -> None:
    logging.info(f"{bot.user} has connected to Discord!")


@bot.command(name="clearcontext")
async def clear_context(ctx):
    """Clear the conversation context for the user"""
    user_id = ctx.author.id
    if user_id in user_contexts:
        del user_contexts[user_id]
        await ctx.send("Your conversation context has been cleared.")
    else:
        await ctx.send("No context to clear.")


@bot.event
async def on_message(message) -> None:
    """Respond only to direct mentions and commands"""
    if message.author == bot.user:
        return

    # Only process messages that mention the bot
    if not (bot.user.mentioned_in(message)):
        return

    if bot.user.mentioned_in(message):
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        if not question:
            await message.reply("Please ask a question after mentioning me.")
            return

        # Get user context
        user_id = message.author.id
        context = user_contexts.get(user_id, [])

        # Get system prompt for the current provider
        system_prompt = llm_handler.get_system_prompt(DEFAULT_LLM_PROVIDER)

        # Use the configured default provider
        if DEFAULT_LLM_PROVIDER == "ollama":
            response = await llm_handler.get_ollama_response(
                question, context, system_prompt
            )
        elif DEFAULT_LLM_PROVIDER in ["openai", "openai-compatible"]:
            response = await llm_handler.get_openai_response(
                question, context, system_prompt
            )
        elif DEFAULT_LLM_PROVIDER == "gemini":
            response = await llm_handler.get_gemini_response(
                question, context, system_prompt
            )
        elif DEFAULT_LLM_PROVIDER == "deepseek":
            response = await llm_handler.get_deepseek_response(
                question, context, system_prompt
            )
        else:
            response = "Invalid LLM provider configured"

        if not response:
            await message.reply(
                "Sorry, I encountered an error processing your request."
            )
            return

        # Update context
        user_contexts[user_id].append({"role": "user", "content": question})
        user_contexts[user_id].append({"role": "assistant", "content": response})

        # Keep only the last MAX_CONTEXT_LENGTH messages
        if len(user_contexts[user_id]) > MAX_CONTEXT_LENGTH * 2:
            user_contexts[user_id] = user_contexts[user_id][-MAX_CONTEXT_LENGTH * 2 :]

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
