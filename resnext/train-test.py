# -*- coding:utf-8 -*-

import sys
sys.path.append("../lib")
from Teacher import Teacher

if __name__ == "__main__":
    teacher = Teacher("d:/project/tianchi/data/", "resnet_idmap_v1/", 4, snapshotFilename="")
    teacher.train()
