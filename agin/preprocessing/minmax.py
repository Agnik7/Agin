import numpy as np
class MinMaxScaler():
    """ 
    A class to perform Min-Max scaling (normalization) on the given data.

    This class scales the feature data to a specified range (default is between 0 and 1). 
    It normalizes each feature by subtracting the minimum value of that feature and 
    then dividing by the range (max - min). The result is then scaled to the desired range.
    """
    def scale(self,data, feature_range=(0, 1)):
        """ 
        Function to scale the input data to a specified feature range using Min-Max scaling.

        This method performs the Min-Max normalization on the input data, transforming the values 
        of each feature (column) to a specific range, typically [0, 1]. The minimum and maximum 
        values of the data are used to calculate the scaling factor, and the scaled values are returned.

        Args: 
            data (numpy array): The input data to be scaled. Each column represents a feature.
            feature_range (tuple, optional): The desired range for scaling. Default is (0, 1).

        Returns:
            numpy array: The scaled data, where each feature is scaled to the specified range.
        """
        # Extract minimum and maximum values of the desired feature range
        min_val = feature_range[0]
        max_val = feature_range[1]

        # Calculate the minimum and maximum values of the input data along each feature (column)
        data_min = np.min(data, axis=0)  # Minimum of each column
        data_max = np.max(data, axis=0)  # Maximum of each column

        # Avoid division by zero by replacing zero ranges with 1
        range_val = data_max - data_min
        range_val[range_val == 0] = 1

        # Apply scaling: First normalize to [0, 1], then scale to the target range
        scaled_data = (data - data_min) / range_val
        scaled_data = scaled_data * (max_val - min_val) + min_val

        return scaled_data