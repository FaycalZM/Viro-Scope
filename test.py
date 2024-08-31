import json


output = json.loads('{"predictions" : [],"actual":[]}')
output["predictions"] = [1, 2, 3]
output["actual"] = [2, 3]


print(output)
