from setuptools import setup, find_packages
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()
print(f"Install Requires = {install_requires}")
setup(
    name="Agin",
    version="0.1.2",
    description="A one-stop machine learning solution",
    author="Agnik Bakshi, Indranjana Chatterjee",
    packages=find_packages(),
    install_requires=install_requires
)
