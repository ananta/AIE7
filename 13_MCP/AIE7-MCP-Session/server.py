from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)


@mcp.tool()
def generate_fun_fact(topic: str) -> str:
    """Generate a random fun fact about the given topic"""
    facts = {
        "space": [
            "A day on Venus is longer than a year on Venus.",
            "Neutron stars are so dense that a sugar-cube-sized piece would weigh about a billion tons."
        ],
        "ocean": [
            "More than 80% of the ocean is unexplored.",
            "The blue whale is the largest animal to have ever existed."
        ],
        "history": [
            "Oxford University is older than the Aztec Empire.",
            "Napoleon was once attacked by a horde of bunnies."
        ]
    }
    topic_facts = facts.get(topic.lower(), ["I don't have facts about that topic yet!"])
    import random
    return random.choice(topic_facts)

import requests

@mcp.tool()
def get_weather(city: str, units: str = "metric") -> str:
    """Get the current weather for a given city. Units can be 'metric', 'imperial', or 'standard'."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set in .env file."
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": units
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if response.status_code != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        name = data["name"]
        temp = data["main"]["temp"]
        weather_desc = data["weather"][0]["description"].capitalize()
        feels_like = data["main"]["feels_like"]
        
        return f"Weather in {name}: {weather_desc}, {temp}° ({'feels like ' + str(feels_like) + '°'})"
    except Exception as e:
        return f"Error fetching weather: {e}"


@mcp.tool()
def ping(msg: str = "pong") -> str:
    """Simple ping tool to test MCP connectivity"""
    return f"Pong! Received: {msg}"


if __name__ == "__main__":
    mcp.run(transport="stdio")


