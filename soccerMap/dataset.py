import glob

class Dataset:
    path = ''

    def __init__(self, path):
        self.path = path

    def getPics(self):
        files = glob.glob(self.path, recursive=True)
        for file in files:
            yield file