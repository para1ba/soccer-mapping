import glob

class Dataset:
    def __init__(self, path):
        self.path = path

    def getPics(self):
        files = glob.glob(self.path + "/**/*.*", recursive=True)
        print(list)
        return list(files)