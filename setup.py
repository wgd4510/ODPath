from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", encoding="utf8") as f:
        return f.read().splitlines()


setup(
    name="odpath",
    packages=find_packages(exclude=[]),
    version="0.0.9",
    license="MIT",
    description="get files path and name",
    author="Guodong Wu",
    author_email="461854822@qq.com",
    long_description_content_type="text/markdown",
    keywords=["file path"],
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
