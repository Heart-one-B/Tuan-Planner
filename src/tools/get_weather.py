from src.utils.config_handler import tools_conf
import requests
from langchain_core.tools import tool

@tool(parse_docstring=True)
def get_weather(city:str)-> str:
    """获取天气的工具，获取当前时刻指定城市的天气信息。

    Args:
        city: 待查询的指定城市。

    Returns:
        指定城市的天气信息，包括温度、天气状况、风向、湿度等。
    """
    api_key = tools_conf['weather_api_key']
    geo_url = f"https://pc3tehqmcy.re.qweatherapi.com/geo/v2/city/lookup?location={city}&key={api_key}"

    try:
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        if geo_data['code'] != '200':
            return f"找不到城市: {city}，请确认名称是否正确。"

        # 获取第一个匹配城市的 ID 和完整名称
        city_id = geo_data['location'][0]['id']
        official_name = geo_data['location'][0]['name']

        # 实时天气 API
        weather_url = f"https://pc3tehqmcy.re.qweatherapi.com/v7/weather/now?location={city_id}&key={api_key}"

        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

        if weather_data['code'] == '200':
            now = weather_data['now']
            temp = now['temp']  # 温度
            text = now['text']  # 天气状况（晴、多云等）
            wind = now['windDir']  # 风向
            humidity = now['humidity']  # 湿度

            result = f"{official_name}当前天气：{text}，温度 {temp}℃，{wind}，湿度 {humidity}%。"
            return result
        else:
            return "获取天气数据失败。"
    except Exception as e:
        return f"查询出错: {str(e)}"
