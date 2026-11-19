#!/usr/bin/env python3
"""One-command build for the supplied A and B companies."""
import argparse, subprocess, sys
p=argparse.ArgumentParser()
p.add_argument('--root',default='Cantill_DataML_Task_Kit',help='directory containing the unzipped private task kit')
p.add_argument('--output-dir',default='outputs')
a=p.parse_args()
for config in ('config_A.json','config_B.json'):
    cmd=[sys.executable,'spend_engine.py','--config',config,'--root',a.root,'--output-dir',a.output_dir]
    subprocess.run(cmd,check=True)
print('Build complete. Outputs:',a.output_dir+'/abhinav__A.csv and abhinav__B.csv')
