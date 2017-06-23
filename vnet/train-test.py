# -*- coding:utf-8 -*-

import sys
sys.path.append("../common")
from Teacher import Teacher

if __name__ == "__main__":
    teacher = Teacher("d:/project/tianchi/data/", "v5/", 4, snapshotFilename="_iter_19000.solverstate")
    teacher.train()
