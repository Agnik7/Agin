# **Utils**

The `utils` module provides utility classes and methods to enhance your project.

## **Health Class**
The `Health` class helps ensure the package is functioning correctly by providing a simple health-check mechanism.

### **Usage**
The `Health` class can be imported directly from the `agin` package or from the `agin.utils` module:

```python
from agin import Health
# or
from agin.utils import Health
```
#### **Example**
```python
# Option 1: Importing directly from agin
from agin import Health

# Option 2: Importing from agin.utils
from agin.utils import Health

# Create a Health object
health = Health("Excellent")

# Check the health status
print(health.check_health())  # Output: Health status: Excellent
```

### **Methods**
#### **`__init__(status="Good")`**
   - Initializes the `Health` class with a default status of "Good".
   - **Args**: `status` (string)

#### **`check_health()`**
   - Returns the current health status.
   - **Args**: None
   - **Returns**: String representing the health status.
