import sys

class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'a+')

    def write(self, m):
        if m == "\n":
            self.log.write(m)
        elif m[:2] == "!#":
            self.log.write(m[2:])
        else:
            self.log.write("# " + m)
        self.terminal.write(m)
        self.log.flush()

    def flush(self):
        self.log.flush()
