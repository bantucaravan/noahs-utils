# all standard library pacakges
from packaging.requirements import Requirement
import packaging.requirements
import subprocess
import shlex
from pathlib import Path
import sys
import re

def parse_reqs(reqs_str):
    '''
    Takes the str contents of reqs file,  returns a parsed 
    requirement object (if non-comment portion of line is parsable as a requirement) and 
    comment (if any) for each line.

    :param reqs_str: string contents of a requirements.txt file
    :returns: reqs --> list of `packaging.requirements.Requirement` (or whatever string was present if requirement (not comment) could not be parsed).
        and comments --> list of comment strings or empty string if no comment on given line.
        Both lists are the same length and the number of lines in the file.
    '''
    #TODO: optionally handle saving comment lines (and commenting out invalid req lines)
    reqs = []
    comments = []
    for r in reqs_str.splitlines():
        i = r.find('#')
        r, c =  (r, '') if (i==-1) else (r[:i], r[i:])  # extract all string after comment char
        try:
            reqs.append(Requirement(r))
        except packaging.requirements.InvalidRequirement as e:
            reqs.append(r)
            import traceback
            print(traceback.format_exception(*sys.exc_info())[-1])
        comments.append(c)
        
    return reqs, comments


##### Add/update versions for existing pacakge specifications in requirements.txt ########
if __name__ == '__main__':
    '''
    Script takes a requirements file and add/updates version specifications for 
    each package based on the versions installed in the active python 
    environment. The script prints the new/updated requirements.txt 
    content to stdout.
    
    Call this script like this: `python parse_requirements.py "path/to/requirements.txt"`
    '''
    # reqs_file = 'requirements.txt'
    reqs_file = sys.argv[1]


    # get reqs from requirements.txt
    reqs_str = Path(reqs_file).read_text()
    reqs, comments = parse_reqs(reqs_str)

    # collect installed pacakge versions from pip freeze
    # pipcmd = f'''"{sys.executable}" -m pip freeze'''
    pipcmd = f'''"{sys.executable}" -m pip list --format=freeze''' # this works with conda envs
    # NB: there is some weird phenomena where pip in some unclear circumstances replaces underscores with dashes in pacakge names
    installed_reqs_str = subprocess.check_output(shlex.split(pipcmd)).decode()
    installed_reqs, _ = parse_reqs(installed_reqs_str)
    # subpackages e.g. "haystack[inference]" seem to be dropped in pip freeze?

    # assign version specifiers (from pip freeze output) for all pacakges in reqs file
    for req in reqs:
        if isinstance(req, Requirement):
            # Q: is it true that package names never differ by case only (are case-insensitive?)
            # NB: it seems that pip replaces any non alphanumeric char with a dash. See: https://stackoverflow.com/questions/19097057/pip-e-no-magic-underscore-to-dash-replacement/19131777#19131777
            req_with_ver = [r for r in installed_reqs if r.name.lower() == re.sub(r"[^a-z0-9]", '-', req.name.lower())]
            if req_with_ver:
                req.specifier = req_with_ver[0].specifier #  update the original req to keep the "extras" (like subpacakges that are dropped in pip freeze outout)
                # MAYBE: consider vars().update()
            else:
                raise Exception(f'{req.name} is in requirements.txt but not in pip freeze output')

    # write updated reqs
    # reqs_with_vers_str = '\n'.join([str(r) for r in reqs])
    reqs_with_vers_str = '\n'.join([f"{str(r)} {c}" if isinstance(r, Requirement) else r+c for r,c in zip(reqs, comments)])
    print(reqs_with_vers_str)
    # Path(reqs_file).write_text(reqs_with_vers_str )


