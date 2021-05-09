# -*- coding: utf-8 -*-
"""
Created on Sun May  9 02:47:39 2021

@author: SULTAN
"""

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Depression(BaseModel):
    Text: str 
    