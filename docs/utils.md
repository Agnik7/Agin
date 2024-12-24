# Utils

The `utils` module provides utility classes and methods to enhance your project.

## Health Class
The `Health` class helps ensure the package is functioning correctly by providing a simple health-check mechanism.

### Methods
1. **`__init__(status="Good")`**
   - Initializes the `Health` class with a default status of "Good".
   - **Args**: `status` (string)

2. **`check_health()`**
   - Returns the current health status.
   - **Args**: None
   - **Returns**: String representing the health status.

### Example
```python
from agin import Health

# Create a Health object
health_check = Health("Excellent")

# Check the health status
print(health_check.check_health())  # Output: Health status: Excellent
