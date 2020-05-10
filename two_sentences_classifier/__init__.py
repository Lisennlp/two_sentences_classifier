import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../two_sentences_classifier')))

from two_sentences_classifier import *