#Convert supr file in a format that is readable for cpp

import numpy as np
import json
import chumpy
import argparse
from pathlib import Path

def converttojson(js, key, value):
    if type(value) == int:
        inttojson(js, key, value)
    elif type(value) == float:
        floattojson(js, key, value)
    elif type(value) == dict:
        dicttojson(js, key, value)
    elif type(value) == np.ndarray:
        nptojson(js, key, value)
    elif type(value) == chumpy.ch.Ch:
        chtojson(js, key, value)
    elif type(value) == float:
        floattojson(js, key, value)
    elif type(value) == str:
        strtojson(js, key, value)
    elif type(value) == list:
        listtojson(js, key, value)
    else:
        deftojson(js, key, value)


def inttojson(js, key, value):
    js[key] = {}
    js[key]["type"] = "int"
    js[key]["data"] = value


def strtojson(js, key, value):
    js[key] = {}
    js[key]["type"] = "string"
    js[key]["data"] = value


def floattojson(js, key, value):
    js[key] = {}
    js[key]["type"] = "float"
    js[key]["data"] = value


def deftojson(js, key, value):
    js[key] = {}
    js[key]["type"] = str(type(value))
    js[key]["data"] = value


def nptojson(js, key, value):
    js[key] = {}
    js[key]["type"] = "array"
    js[key]["shape"] = value.shape
    js[key]["data"] = value.flatten().tolist()
    if value.dtype == np.int64:
        js[key]["datatype"] = "int"
    elif value.dtype == np.float64:
        js[key]["datatype"] = "float"
    elif value.dtype == np.uint32:
        js[key]["datatype"] = "uint"


def chtojson(js, key, value):
    nptojson(js, key, np.array(value))


def listtojson(js, key, value):
    nptojson(js, key, np.array(value))


def dicttojson(js, key, value):
    js[key] = {}
    js[key]["type"] = "dict"
    vnew = {}
    for k, vs in value.items():
        converttojson(vnew, k, vs)
    js[key]["data"] = vnew

parser = argparse.ArgumentParser()
parser.add_argument("-p",type=str,default="./supr_neutral.npy")
args = parser.parse_args()


file = args.p
np.savez('./'+ Path(file).stem + '.npz',**{k: np.ascontiguousarray(v) for k, v in np.load(file, allow_pickle=True, encoding='latin1').tolist().items() if k in {'f', 'J', 'J_regressor', 'kintree_table', 'posedirs', 'shapedirs', 'v_template', 'weights'}})
loadedlist = np.load(file, allow_pickle=True, encoding='latin1').tolist().items()
print(file)
json_opbject = json.loads('{}')
for k, v in loadedlist:
    if k in {'f', 'J', 'J_regressor', 'kintree_table', 'posedirs', 'shapedirs', 'v_template', 'weights'}:
        continue
    converttojson(json_opbject, k, v)
print("Writing begins now")
with open('./'+ Path(file).stem + ".json", "w") as ofile:
    json.dump(json_opbject, ofile, indent=4, sort_keys=True)
