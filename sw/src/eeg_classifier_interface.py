import pylsl
import time
import random

info = pylsl.stream_info('Classifier_Result_Out', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream')
result_outlet = pylsl.stream_outlet(info, 1, 1)

print("ready")

while True:
    ans = ['left','right'][random.randint(0,1)]
    print(f'i classified: {ans}')
    result_outlet.push_sample(pylsl.vectorstr([ans]))
    time.sleep(2)