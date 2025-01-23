from typing import List

from model_cofig.config import RECONSTRUCTED_BASE_SENT
from process_model.model.Entity import Entity


# 用作存储re任务的从文章中抽出出来的基本格式，单位为一个句子一个对象
class ReSentBaseData:
    def __init__(self, sent: str, head_entity: Entity, tail_entity: Entity, relation_type: str,
                 sent_token_list: List[str]):
        """
        抽取出文本中需要进行关系抽取的句子
        :param sent:
        :param head_entity:
        :param tail_entity:
        :param relation_type:
        :param sent_token_list:
        """
        self.sent = sent # type: str
        self.head_entity = head_entity # type: Entity
        self.tail_entity = tail_entity # type: Entity
        self.relation_type = relation_type # type: str
        self.sent_token_list = sent_token_list # type: List[str]
        self.reconstructed_sent = self._reconstructed_sent() # type: str
        self.fine_tuned_re_model_tokens = self._get_reconstruct_sent_token_list() # type: List[str]

    def to_dict(self) -> dict:
        return {
            'sent': self.sent,
            'head_entity': self.head_entity.to_dict(),
            'tail_entity': self.tail_entity.to_dict(),
            'relation_type': self.relation_type,
            'sent_token_list': self.sent_token_list,
        }

    @staticmethod
    def from_dict(data):
        head_entity = Entity.from_dict(data['head_entity'])
        tail_entity = Entity.from_dict(data['tail_entity'])
        return ReSentBaseData(data['sent'], head_entity, tail_entity, data['relation_type'], data['sent_token_list'])

    def _get_reconstruct_sent_token_list(self) -> List[str]:
        reconstruct_sent_token_list = []
        i = 0
        head_len = len(self.head_entity.entity_name_list)
        tail_len = len(self.tail_entity.entity_name_list)

        while i < len(self.sent_token_list):
            if i + head_len <= len(self.sent_token_list) and self.sent_token_list[
                                                             i:i + head_len] == self.head_entity.entity_name_list:
                reconstruct_sent_token_list.append(f"[OBJ_{self.head_entity.entity_type.upper()}]")
                reconstruct_sent_token_list.extend(self.sent_token_list[i:i + head_len])
                reconstruct_sent_token_list.append(f"[/OBJ_{self.head_entity.entity_type.upper()}]")
                i += head_len
            elif i + tail_len <= len(self.sent_token_list) and self.sent_token_list[
                                                               i:i + tail_len] == self.tail_entity.entity_name_list:
                reconstruct_sent_token_list.append(f"[SUB_{self.tail_entity.entity_type.upper()}]")
                reconstruct_sent_token_list.extend(self.sent_token_list[i:i + tail_len])
                reconstruct_sent_token_list.append(f"[/SUB_{self.tail_entity.entity_type.upper()}]")
                i += tail_len
            else:
                reconstruct_sent_token_list.append(self.sent_token_list[i])
                i += 1

        return reconstruct_sent_token_list

    def _reconstructed_sent(self) -> str:
        head_entity = self.head_entity.entity_name
        tail_entity = self.tail_entity.entity_name
        sent = self.sent
        return RECONSTRUCTED_BASE_SENT.format(head_entity=head_entity, tail_entity=tail_entity, input_sent=sent)