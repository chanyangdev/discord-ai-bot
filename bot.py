import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_ai_api_key():
    if GEMINI_API_KEY:
        return GEMINI_API_KEY, "gemini"
    if GROQ_API_KEY:
        return GROQ_API_KEY, "groq"
    return None, "none"


AI_API_KEY, AI_PROVIDER = get_ai_api_key()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"AI provider: {AI_PROVIDER}")


@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")


bot.run(os.getenv("DISCORD_BOT_TOKEN"))


