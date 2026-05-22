from pprint import pprint
from src.agent.intent_agent import IntentAgent

result = IntentAgent().parse(
  "今天一整天想和闺蜜出去玩一下，上午想逛街，下午想喝咖啡，朋友不能吃辣，别太远",
  "北京"
)

pprint(result)