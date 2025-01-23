class EntityTuple:
    def __init__(self, entity_type_name: str, entity_type_value: int):
        self.entity_type_name = entity_type_name
        self.entity_type_value = entity_type_value

    def get_enum_name(self) -> str:
        return self.entity_type_name

    def get_enum_value(self) -> int:
        return self.entity_type_value


# 定义枚举类
class EntityType:
    OTHER_ENTITY_ID = EntityTuple("其他实体", 0)
    HEAD_ENTITY_ID = EntityTuple("头实体", 1)
    TAIL_ENTITY_ID = EntityTuple("尾实体", 2)
