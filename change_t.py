from src.graph.nodes import _estimate_candidate_route_minutes
from src.tools.mock_api import MockToolAPI


candidate = {
  "id": "plan_1",
  "steps": [
      {
          "poi_type": "activity",
          "poi_id": "A1",
      },
      {
          "poi_type": "restaurant",
          "poi_id": "R1",
      },
  ],
  "activities": [
      {
          "id": "A1",
          "name": "活动A",
          "coordinates": "116.397128,39.916527",
      },
  ],
  "restaurants": [
      {
          "id": "R1",
          "name": "餐厅R",
          "coordinates": "116.407128,39.926527",
      },
  ],
}

result = _estimate_candidate_route_minutes(
  candidate=candidate,
  origin_coordinates="116.387128,39.906527",
  api=MockToolAPI(),
)

print(result)