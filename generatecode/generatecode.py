
import yaml
from yaml import load

with open("/home/kalyan/02_gitrepo/Python_Project/generatecode/config.yaml", "r") as stream:
    try:
        from yaml import CLoader as Loader
        print(yaml.safe_load(stream))
        dictionary = yaml.load(stream)
    except yaml.YAMLError or ImportError as exc:
        from yaml import Loader
        print(exc)

'''
for key, value in dictionary.items():
    print (key + " : " + str(value))
    filename = value


code = """
def helloworld(name):
    print(f"hello, {name}!")

helloworld("test")
"""

# Open the file in write mode
with open(filename, "w") as file:
    # Write the Python code to the file
    file.write(code)

print(f"{filename} created successfully.") '''
