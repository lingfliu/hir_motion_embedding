
class BaseLoader:

    """
    加载骨骼hierarchy与其他数据
    """
    def load_meta_all(self, root_dir):
        pass

    """
    加载所有骨骼数据
    """
    def load_all(self, root_dir):
        pass

    """
    load single skeleton
    """
    def load(self, file_path):
        pass

    def load_meta(self, file_path):
        pass

    """
    加载已解析的数据
    """
    def load_parsed_all(self, root_dir, *args):
        pass

    def load_parsed(self, file_path, *args):
        pass