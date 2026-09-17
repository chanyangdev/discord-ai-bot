import asyncio
import os
from collections import defaultdict, deque

import discord
from dotenv import load_dotenv
from google import genai

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN is missing from .env")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

gemini = genai.Client(api_key=GEMINI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True

discord_client = discord.Client(intents=intents)

MAX_TURNS = 8
conversation_history = defaultdict(lambda: deque(maxlen=MAX_TURNS))


def remove_bot_mention(message: discord.Message) -> str:
    text = message.content

    for mention_format in (
        f"<@{discord_client.user.id}>",
        f"<@!{discord_client.user.id}>",
    ):
        text = text.replace(mention_format, "")

    return text.strip()


def build_contents(thread_id: int, prompt: str) -> list[dict]:
    contents = []

    for turn in conversation_history[thread_id]:
        contents.append(
            {
                "role": "user",
                "parts": [{"text": turn["user"]}],
            }
        )
        contents.append(
            {
                "role": "model",
                "parts": [{"text": turn["assistant"]}],
            }
        )

    contents.append(
        {
            "role": "user",
            "parts": [{"text": prompt}],
        }
    )
    return contents


def ask_gemini(thread_id: int, prompt: str) -> str:
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_contents(thread_id, prompt),
    )
    return response.text or "I couldn't generate a response."


async def send_long_message(channel, text: str):
    # Discord messages have a 2,000-character limit.
    for start in range(0, len(text), 1900):
        await channel.send(text[start : start + 1900])


async def answer_in_thread(thread: discord.Thread, prompt: str):
    try:
        async with thread.typing():
            answer = await asyncio.to_thread(
                ask_gemini,
                thread.id,
                prompt,
            )

        conversation_history[thread.id].append(
            {
                "user": prompt,
                "assistant": answer,
            }
        )

        await send_long_message(thread, answer)

    except Exception as error:
        print(f"Gemini request failed: {error}")
        await thread.send(
            "Sorry, I couldn't contact the AI service. Check the bot terminal for the error."
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

    if isinstance(message.channel, discord.Thread):
        if message.channel.owner_id != discord_client.user.id:
            return

        prompt = remove_bot_mention(message)
        if not prompt:
            return

        await answer_in_thread(message.channel, prompt)
        return

    if discord_client.user not in message.mentions:
        return

    prompt = remove_bot_mention(message)

    if not prompt:
        await message.reply(
            "Mention me with a question and I'll open a thread.",
            mention_author=False,
        )
        return

    try:
        thread_name = prompt.replace("\n", " ")[:80] or "AI conversation"
        thread = await message.create_thread(
            name=thread_name,
            auto_archive_duration=60,
        )
        await answer_in_thread(thread, prompt)

    except discord.Forbidden:
        await message.reply(
            "I need permission to create and send messages in threads.",
            mention_author=False,
        )
    except Exception as error:
        print(f"Thread creation failed: {error}")
        await message.reply(
            "Sorry, I couldn't create a thread. Check the bot terminal for the error.",
            mention_author=False,
        )


discord_client.run(DISCORD_BOT_TOKEN)

