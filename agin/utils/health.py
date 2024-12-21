class Health():
    def __init__(self, status="Good"):
        self.status = status
    def check_health(self):
        """ 
        Function to check the health of the package

        Args: None
        
        Returns:
            String: Returns health status to make sure it is working.
        """
        return f"Health status: {self.status}"
