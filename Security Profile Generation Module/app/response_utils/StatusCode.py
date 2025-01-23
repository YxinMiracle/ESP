from enum import Enum

class StatusCode(Enum):
    SYSTEM_ERROR = (400, "系统错误")
    PARAMETER_ERROR = (410, "参数错误")
    NO_AUTH = (420, "无权访问")
    SUCCESS = (0, "执行成功")

    def __init__(self, code, message):
        self.code = code
        self.message = message
