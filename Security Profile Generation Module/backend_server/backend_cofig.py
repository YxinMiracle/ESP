"""
这里是用于记录后端的一些配置，例如主域名，端口号之类的数据
"""

# headers 字典
BASE_HEADERS = {
    "inner-req-signature": "askliyij%kkefoo)))Yxi0==@@cdfMiracle",
    "Content-Type": "application/json"
}

# headers请求加密信息
AUTH_SECRET_KEY = "alsapYxinMiracleAuthKey"
AUTH_SECRET = "alsapYxinMiracleSecretKeyYep"

# HOST = "http://172.26.211.224"
HOST = "https://alsap.tracesec.cn"
PORT = 8763

# 根据上方两个变量得出的整体路径
BACKEND_URL = HOST + ":" + str(PORT)

# ========================================= ttp相关 =======================================
TTP_SERVER_PATH_PREFIX = "/api/inner/ttp"
ADD_CTI_TTP_URL = BACKEND_URL + TTP_SERVER_PATH_PREFIX + "/add"

# ========================================= cti相关 =======================================
ENTITY_SERVER_PATH_PREFIX = "/api/inner/cti"
ADD_CTI_ENTITY_URL = BACKEND_URL + ENTITY_SERVER_PATH_PREFIX + "/add/entity"
ADD_CTI_CHUNK_URL = BACKEND_URL + ENTITY_SERVER_PATH_PREFIX + "/add/chunk"
UPDATE_CTI_CONTENT_URL = BACKEND_URL + ENTITY_SERVER_PATH_PREFIX + "/update/content"

# ========================================= item相关 =======================================
ITEM_SERVER_PATH_PREFIX = "/api/inner/item"
GET_ITEM_ID_URL = BACKEND_URL + ITEM_SERVER_PATH_PREFIX

# ========================================= entity相关 =======================================
ENTITY_SERVER_PATH_PREFIX = "/api/inner/entity"
ADD_UNIQUE_ENTITY = BACKEND_URL + ENTITY_SERVER_PATH_PREFIX + "/unique/add"
GET_UNIQUE_ENTITY_ID = BACKEND_URL + ENTITY_SERVER_PATH_PREFIX + "/unique/entity_id"

# ========================================= relation_type相关 =======================================
RELATION_TYPE_SERVER_PATH_PREFIX = "/api/inner/relation/type"
IS_RELATION_TYPE_URL = BACKEND_URL + RELATION_TYPE_SERVER_PATH_PREFIX + "/is_relation"

# ========================================= relation相关 =======================================
RELATION_SERVER_PATH_PREFIX = "/api/inner/relation"
ADD_RELATION_URL = BACKEND_URL + RELATION_SERVER_PATH_PREFIX + "/add"

