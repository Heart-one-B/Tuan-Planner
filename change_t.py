from pprint import pprint
import importlib

import src.graph.nodes as nodes_module
import src.tools.mock_api as mock_api_module

importlib.reload(mock_api_module)
importlib.reload(nodes_module)

from src.graph.nodes import restaurant_search_node

state = {
  "constraints": {
      "scenario": "friends",
      "time_window": "today_afternoon",
      "date_label": "今天",
      "daypart": "下午",
      "time_phrase": "今天下午",
      "origin_area": "area_central",
  },
  "constraint_build": {
      "query_constraints": {
          "need_activity": False,
          "need_restaurant": True,
          "time_window": "today_afternoon",
          "keywords_restaurant": ["川菜"],
          "exclude_keywords_restaurant": [],
      }
  },
  "runtime_origin_area": "北京",
  "runtime_origin_coordinates": "",
  "errors": [],
}

restaurant_result = restaurant_search_node(state)

pprint([
  (
      item.get("id"),
      item.get("name"),
      item.get("source"),
      item.get("search_mode"),
      item.get("address"),
      item.get("rating"),
  )
  for item in restaurant_result["restaurants"]
])