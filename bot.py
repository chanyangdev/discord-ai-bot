import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
	print(f"Logged in as {bot.user}")

@bot.command()
async def ping(ctx):
	await ctx.send("Pong!")

bot.run(os.getenv("DISCORD_BOT_TOKEN"))


