# test case: git pull when there are no diffs in a given file relative to HEAD but there are relative to working tree (i.e. local file modified but not committed)

# set up
git init

echo "change 1" > file1.txt