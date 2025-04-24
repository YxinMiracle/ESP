from typing import List


# 用做存储对象基本信息
class Entity:
    def __init__(self, entity_name: str, entity_type: str):
        self.entity_name = entity_name # type: str
        self.entity_type = entity_type # type: str
        self.entity_name_list = self._get_entity_name_list(entity_name) # type: List[str]

    def __str__(self):
        print(f"name={self.entity_name},type={self.entity_type}")

    def to_dict(self) -> dict:
        return {
            'entity_name': self.entity_name,
            'entity_type': self.entity_type,
        }

    @staticmethod
    def from_dict(data):
        return Entity(data['entity_name'], data['entity_type'])

    def _get_entity_name_list(self, entity_name: str) -> List[str]:
        return entity_name.split(" ")
