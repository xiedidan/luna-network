import os
import time

class Logger(object):
    # constructor
    def __init__(self, levelStr, path, name):
        self.LogLevel = {
            "Debug": 0,
            "Info": 1,
            "Warning": 2,
            "Error": 3
        }

        self.level = self.LogLevel[levelStr]
        self.path = path
        self.name = name

        self.logFile = self.path + self.name + ".log"
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.file = open(self.logFile, "a")

    # destructor
    def __del__(self):
        self.file.close()

    # helper
    def log(self, level, source, message):
        ts = time.strftime("%Y-%m-%d %X", time.localtime(time.time()))
        logMsg = ts + " [" + list(self.LogLevel.keys())[list(self.LogLevel.values()).index(level)] + "] from " + source + ": " + message + "\r\n"

        try:
            self.file.write(logMsg)
            self.file.flush()
        except:
            self.file.close()
            self.file = open(self.logFile, "a")

    # interface
    def debug(self, source, message):
        self.log(self.LogLevel["Debug"], source, message)

    def info(self, source, message):
        self.log(self.LogLevel["Info"], source, message)

    def warning(self, source, message):
        self.log(self.LogLevel["Warning"], source, message)

    def error(self, source, message):
        self.log(self.LogLevel["Error"], source, message)
