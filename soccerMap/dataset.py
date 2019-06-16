import glob

class Dataset:
    path = ''

    def __init__(self, path):
        self.path = path

    def getPics(self):
        files = glob.glob(self.path + "/**/*.bmp", recursive=True)
        return list(files)
        