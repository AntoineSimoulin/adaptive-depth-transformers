import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="act",
    version="0.0.1",
    author="Antoine Simoulin",
    author_email="antoine.simoulin@gmail.com",
    description="Adaptive Depth Transformers implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntoineSimoulin/adaptive-depth-transformers",
    packages=['act'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)