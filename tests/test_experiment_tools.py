import git

import shutil
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).parents[1]))
from noahs.experiment_tools import git_snapshot

# path=_dh[0]
# path="/Users/noah.chasek-macfoy@ibm.com/Desktop/projects/Noah's Utils [git repo]/tests"
path = Path(__file__).parent
type(Path(path).parent)

def repo_teardown():
    try:
        repo = git.Repo(path=path)
        shutil.rmtree(repo.git_dir)
    except git.InvalidGitRepositoryError as e:
        print(f'no repo at {path}!!')
        pass

def repo_setup():
    repo_teardown()
    repo = git.Repo.init(path=path)
    return repo
    


def test_git_snapshot():
    # behavior/errors to test:
    # - if base file is not tracked previously
    # - if there are no modification to commit
    # - if commit hash changes after git snapshot
    repo = repo_setup()
    test_file_a = Path('_testa.txt')
    test_file_b = Path('_testb.txt')
    test_file_a.write_text('mod 1')
    test_file_b.write_text('mod 1')
    repo.index.add([str(test_file_a), str(test_file_b)])
    repo.index.commit('setup commit')
    test_file_a.write_text('mod 2')
    sha = git_snapshot(path=str(test_file_a), run_id=1)
    diffs = repo.head.commit.diff(None)
    assert len(diffs) == 0

    # teardown
    test_file_a.unlink()
    test_file_b.unlink()
    repo_teardown()
    




