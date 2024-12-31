<div align="center">
  <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FAgnik7%2FAgin&countColor=%23dce775&style=flat" alt="Agin Visitors"/>
  <img src="https://img.shields.io/pypi/dm/AginPy" alt="PyPI - Downloads"/>
  <img src="https://img.shields.io/github/issues/Agnik7/Agin" alt="GitHub Issues"/>
  <img src="https://img.shields.io/github/issues-pr/Agnik7/Agin" alt="GitHub Pull Requests"/>
  <img src="https://img.shields.io/pypi/l/AginPy" alt="PyPI - License"/>
  <img src="https://img.shields.io/github/watchers/Agnik7/AginPy" alt="GitHub watchers"/>
  <img src="https://img.shields.io/pypi/v/AginPy" alt="PyPI - Version"/>
  <img src="https://img.shields.io/github/commit-activity/t/Agnik7/Agin/main" alt="GitHub commit activity (branch)"/>
  <img src="https://img.shields.io/github/contributors/Agnik7/Agin" alt="GitHub contributors"/>
</div>
<br/>
<div align="center">
  <img src="/assets/LOGO.png" alt="AGIN LOGO FOR README" height="500" width="500"/>
</div>

# Agin

**Agin** is a one-stop machine learning solution designed to streamline your ML workflows with easy-to-use utilities and a flexible structure.

## **Features**
- Modular design for efficient machine learning workflows.
- Easy integration with existing Python projects.
- Simple and intuitive API.

---

## **Installation**

To install the package, use:
```bash
pip install AginPy
```
## **Usage**

Once installed, you can import and use the library as follows:

```python
from agin import Health

# Create an instance of the Health class
h = Health()

# Check health status
print(h.check_health())  # Output: Health status: Good
```
---

## **Setting up the Project Locally**
### **Step 1: Clone the Repository**
Clone this repository to your local machine:

```bash
git clone https://github.com/Agnik7/Agin.git
cd Agin
```

### **Step 2: Set Up a Virtual Environment**
It's recommended to use a virtual environment for managing dependencies. Below are instructions for setting up a virtual environment on different platforms.

#### **Windows**
1. Open a terminal (Command Prompt or PowerShell).
2. Run the following commands:
   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

#### **Linux/MacOS**
1. Open a terminal.
2. Run the following commands:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

To deactivate the virtual environment (on any platform), simply run:
```bash
deactivate
```

### **Step 3: Install the Package in Editable Mode**
Install the package in **editable mode** to allow for modifications during development:

```bash
pip install -e .
```

---

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description of your changes.

---

## **License**
This project is licensed under the GNU General Public License. See the `LICENSE` file for details.
