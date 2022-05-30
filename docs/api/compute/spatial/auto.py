import os


def content(file_name):
    module_name = file_name[:-4] + " module"
    h_len = len(module_name)
    return f"""{module_name}
{"=" * h_len}

.. automodule:: {file_name[:-4]}
    :members:
    :undoc-members:
    :show-inheritance:
    :private-members:
"""


blacklist = ["__init__.py", "__pycache__", "auto.py"]

for file in os.listdir():
    if file not in blacklist:
        f = open(file, "w")
        f.write(content(file))
