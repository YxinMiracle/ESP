from process_model.model.ReBaseData import ReSentBaseData


class IdentifiedReSentBaseData(ReSentBaseData):
    def __init__(self, *args, id: int, **kwargs):
        """
        继承ReSentBaseData，并添加一个唯一的标识符id
        :param args: 传递给ReSentBaseData的位置参数
        :param id: 唯一标识符
        :param kwargs: 传递给ReSentBaseData的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.id = id
