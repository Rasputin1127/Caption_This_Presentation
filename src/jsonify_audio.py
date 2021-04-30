import os
import json
import argparse
import random
import pydub


def get_data(args):
    paths = []
    y = []
    for item in os.walk(args.directory):
        if item[2]:
            for rec in item[2]:
                if rec.endswith('.wav'):
                        dur = float(mediainfo(item[0] + '/' + rec)['duration'])
                        if dur > 15:
                            pass
                        else:
                            path = item[0] + '/' + rec
                            paths.append(path)
                if rec.endswith('.txt'):
                    file = open(f'{item[0]}/' + rec)
                    obj = file.readlines()
                    for _ in obj:
                        text = _.split(maxsplit=1)[1].rstrip('\r\n')
                        y.append(text)

    return paths, y

def jsonify_wavs(filepaths, labels, percent=0.20):
    combine = []
    for i,item in enumerate(zip(file_paths, labels)):
        d = {"key":item[0],"text":item[1]}
        combine.append(dict(d))
    random.shuffle(combine)
    
    with open('train.json','w') as f:
        d = len(combine)
        i = 0
        while i<int(d-d*percent):
            dump = combine[i]
            line = json.dumps(dump)
            f.write(line + "\n")
            i += 1

    with open('test.json','w') as f:
        d = len(combine)
        i = int(d-d*percent)
        while i<d:
            dump = combine[i]
            line = json.dumps(dump)
            f.write(line + "\n")
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for making json files from wav directories.")
    parser.add_argument('--directory', type=str, default=None, required=True
                        help='top-level directory where wav and text files are stored')
    args = parser.parse_args()

    file_paths, labels = get_data(args)
    jsonify_wavs(file_paths, labels)