# Feature 5: Tool - marine weather
# Flow: turn the place name into coordinates (Open-Meteo geocoding) -> fetch the
# current wave height, period and direction (Open-Meteo marine API) -> return it as text.
# Free API, no key needed.

import httpx

SCHEMA = {
    "name": "get_marine_weather",
    "description": "Current wave height, period and direction for a coastal place name.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "A coastal city or beach name"}},
        "required": ["location"],
    },
}

async def run(location: str) -> str:
    async with httpx.AsyncClient(timeout=10) as http:
        g = (await http.get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": location, "count": 1})).json()

        if not g.get("results"):
            return f"Could not find a place called {location}."
        p = g["results"][0]
        m = (await http.get("https://marine-api.open-meteo.com/v1/marine",
                            params={"latitude": p["latitude"], "longitude": p["longitude"],
                                    "current": "wave_height,wave_period,wave_direction"})).json()
        if "current" not in m:
            return f"No marine data for {p['name']}: {m.get('reason', 'unavailable')}"
        return f"{p['name']}: {m['current']}"