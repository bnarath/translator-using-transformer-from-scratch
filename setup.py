from setuptools import setup, find_packages

setup(
    name="translater",
    version="0.1.0",
    packages=find_packages(where="translater"),
    package_dir={"": "translater"},
    install_requires=open("requirements.txt").read().splitlines(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
