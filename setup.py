import setuptools


setuptools.setup(
    name="noahs",
    version="0.0.1",
    author="Noah Chasek-Macfoy",
    author_email="bantucaravan@gmail.com",
    description="Data Science tools I like to use.",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)