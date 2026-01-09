#!/bin/bash

python test_tp_qwen30b_2.py --tp 1 --output result_detailed_tp1_run1.xlsx
python test_tp_qwen30b_2.py --tp 2 --output result_detailed_tp2_run1.xlsx
# python test_tp_qwen30b_2.py --tp 2 --output result_detailed_tp2_run1.xlsx
python test_tp_qwen30b_2.py --tp 4 --output result_detailed_tp4_run1.xlsx
# python test_tp_qwen30b_2.py --tp 4 --output result_detailed_tp4_run1.xlsx