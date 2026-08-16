import asyncio, json
from agents.fact.tools import FactToolset

async def main():
    ts = FactToolset()
    await ts.geocode("四川大学江安校区", city="成都")
    for flag in (True, False):
        d = json.loads(await ts.search_pois("火锅", is_restaurant=flag))
        print(f"is_restaurant={flag} → {d['count']} 家")

asyncio.run(main())