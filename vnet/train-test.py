# -*- coding:utf-8 -*-

import sys
sys.path.append("../common")
from Teacher import Teacher

if __name__ == "__main__":
    teacher = Teacher("d:/project/tianchi/data/", "v5/", 1, snapshotFilename="")
    teacher.train()
