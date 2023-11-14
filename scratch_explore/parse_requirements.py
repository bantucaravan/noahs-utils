
from packaging.requirements import Requirement
import packaging.requirements
import subprocess
import shlex
from pathlib import Path
import sys

def parse_reqs(reqs_str):
    reqs = []
    for r in reqs_str.splitlines():
        try:
            reqs.append(Requirement(r))
        except packaging.requirements.InvalidRequirement as e:
            print(type(e), e)
    return reqs


##### Add versions for existing reqs.txt ########

reqs_file = 'requirements.txt'


# get reqs from requirements.txt
reqs_str = Path(reqs_file).read_text()
reqs = parse_reqs(reqs_str)

# get installed reqs from pip freeze
pipcmd = f'''"{sys.executable}" -m pip freeze'''
pipcmd = f'''"{sys.executable}" -m pip list --format=freeze''' # this works with conda envs
installed_reqs_str = subprocess.check_output(shlex.split(pipcmd)).decode()
installed_reqs = parse_reqs(installed_reqs_str)
# subpackages e.g. "haystack[inference]" seem to be dropped in pip freeze?

# get version specifiers (from pip output) for all pacakges in reqs file
# Q: is it true that package names never differ by case only (are case-insensitive?)
installed_req_names = [i.name.lower() for i in installed_reqs]
get_req = lambda name: [i for i in installed_reqs if i.name.lower() == name.lower()][0]
reqs_with_vers = []
for req in reqs:
    if req.name.lower() in installed_req_names:
        reqs_with_vers.append(get_req(req.name))
    else:
        raise Exception(f'{req.name} is in requirements.txt but not in pip output')

# write updated reqs
# reqs_with_vers_str = '\n'.join([str(r) for r in reqs_with_vers])
reqs_with_vers_str = '\n'.join([str(r.name)+str(r.specifier) for r in reqs_with_vers]) # just name and version
print(reqs_with_vers_str)
# Path(reqs_file).write_text(reqs_with_vers_str )


