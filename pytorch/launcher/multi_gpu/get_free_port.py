#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-04-30)

import sys
import os

subtools = os.getenv('SUBTOOLS')
sys.path.insert(0, '{}/pytorch'.format(subtools))

import libs.support.utils as utils

print(utils.get_free_port())
