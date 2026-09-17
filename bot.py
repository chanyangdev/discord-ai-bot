import asyncio
import os

import discord
from dotenv import load_dotenv
from google import genai

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN is missing from .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

gemini = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

discord_client = discord.Client(intents=intents)


def ask_gemini(prompt: str) -> str:
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text or "I couldn't generate a response."


async def send_long_message(message: discord.Message, text: str):
    # Discord messages have a 2,000-character limit.
    for start in range(0, len(text), 1900):
        await message.reply(
            text[start : start + 1900],
            mention_author=False,
        )


@discord_client.event
async def on_ready():
    print(f"Logged in as {discord_client.user}")
    print(f"Using Gemini model: {GEMINI_MODEL}")


@discord_client.event
async def on_message(message: discord.Message):
    # Never respond to bots, including itself.
    if message.author.bot:
        return

    # Only respond when this bot is mentioned.
    if discord_client.user not in message.mentions:
        return

    # Remove the bot mention from the prompt.
    prompt = message.content

    for mention_format in (
        f"<@{discord_client.user.id}>",
        f"<@!{discord_client.user.id}>",
    ):
        prompt = prompt.replace(mention_format, "")

    prompt = prompt.strip()

    if not prompt:
        await message.reply(
            "Hi! Mention me with a question.",
            mention_author=False,
        )
        return

    try:
        async with message.channel.typing():
            # Run the synchronous Gemini request without freezing Discord.
            answer = await asyncio.to_thread(ask_gemini, prompt)

        await send_long_message(message, answer)

    except Exception as error:
        print(f"Gemini request failed: {error}")
        await message.reply(
            "Sorry, I couldn't contact the AI service. Check the bot terminal for the error.",
            mention_author=False,
        )


discord_client.run(DISCORD_BOT_TOKEN)

