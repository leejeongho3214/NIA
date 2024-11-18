# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
from logging import StreamHandler, Handler, getLevelName
from utils import mkdir
from datetime import datetime


# this class is a copy of logging.FileHandler except we end self.close()
# at the end of each emit. While closing file and reopening file after each
# write is not efficient, it allows us to see partial logs when writing to
# fused Azure blobs, which is very convenient
class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        """
        Open the specified file and use it as the stream for logging.
        """
        # Issue #27493: add support for Path objects to be passed in
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            Handler.__init__(self)
            self.stream = None
        else:
            StreamHandler.__init__(self, self._open())

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                StreamHandler.close(self)
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        return open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        if self.stream is None:
            self.stream = self._open()
        StreamHandler.emit(self, record)
        self.close()

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)


def setup_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    today = datetime.today()

    year = today.year
    month = today.month
    day = today.day
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter1 = logging.Formatter('\r\033[95m'+"%(asctime)s\033[0m  %(message)s") 
    formatter2 = logging.Formatter("%(asctime)s %(message)s")
    ch.setFormatter(formatter1)
    logger.propagate = False
    logger.addHandler(ch)
    filename = f"{year}-{month}-{day}.txt"
    
    if not os.path.isdir(path):
        mkdir(path)
        
    fh = FileHandler(os.path.join(path, filename), encoding='utf-8', mode = "at")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter2)
    logger.addHandler(fh)
    
    return logger
