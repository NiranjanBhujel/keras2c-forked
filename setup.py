import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras2c",
    version="0.0.1",
    author="Niranjan Bhujel",
    author_email="niranjan.bhujel2014@gmail.com",
    description="Package to generate C code for tensorflow model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "https://github.com/NiranjanBhujel/tf2c/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={
        "": ["data/*"],
    },
    python_requires=">=3.6",
)
