# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pdf2image import convert_from_path
import cv2
import matplotlib.pyplot as plt

jaypopplerpath = r"C:\Users\Jay\repos\SewAligned\poppler-20.12.1\Library\bin"

image = convert_from_path('cal_tool.pdf', size=(4000, None), poppler_path=jaypopplerpath)
