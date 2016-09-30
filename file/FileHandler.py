import json

def write_json(file_name, json_data):
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile, sort_keys=True)
        outfile.close()

def read_json(file_name):
    with open(file_name, 'r') as infile:
        data = json.load(infile)
        infile.close()
        return data
