import re
import json

pattern = re.compile(r"\{.*?\}")
text_file = 'experiments/kitti/naiter/lccraft_large/log/test_iterative_1_2025-02-02-09-17-20.log'
with open(text_file,'r') as f:
    text = f.readlines()[-5:]
text = ''.join(text)
results = pattern.findall(text)
time_list = []
if results:
    for result in results:
        result_json = re.sub(r"'",r'"',result)
        time_list.append(json.loads(result_json)['time'])
    print("%0.2f"%(sum(time_list)/len(time_list)*1000))
else:
    print("no match")