class RagMetricsObject:
    """
    Base class for RagMetrics objects that can be stored on the platform.
    
    This abstract class provides common functionality for objects that can be
    serialized to and from the RagMetrics API, including saving, downloading,
    and conversions between Python objects and API representations.
    
    All RagMetrics object classes (Dataset, Criteria, etc.) inherit from this class.
    """

    object_type: str = None

    def to_dict(self):
        """
        Convert the object to a dictionary representation.
        
        This method must be implemented by subclasses to define how the object
        is serialized for API communication.

    
        Returns:
            dict: Dictionary representation of the object.
            
    
        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create an object instance from a dictionary.
        
        This method creates a new instance of the class from data received
        from the API. Subclasses may override this to customize deserialization.

    
        Args:
            data: Dictionary containing object data.

    
        Returns:
            RagMetricsObject: A new instance of the object.
        """
        # Basic implementation assumes constructor args match dict keys
        # Subclasses might need to override for more complex deserialization
        return cls(**data)

    def save(self):
        """
        Save the object to the RagMetrics API.
        
        This method sends the object to the RagMetrics API for storage and
        retrieves the assigned ID.

    
        Returns:
            Response: The API response from saving the object.
            
    
        Raises:
            ValueError: If object_type is not defined.
            Exception: If the API request fails.
        """
        from ragmetrics.api import ragmetrics_client, RagMetricsAuthError, RagMetricsAPIError # Import exceptions

        # Check auth *before* doing anything else
        if not ragmetrics_client.access_token:
            raise RagMetricsAuthError("RagMetrics client not authenticated. Please login first.")
        if not self.object_type:
            raise ValueError("object_type must be defined for the class.")
        
        payload = self.to_dict()
        endpoint = f"/api/client/{self.object_type}/save/"
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        
        response = ragmetrics_client._make_request(
            method="post", endpoint=endpoint, json=payload, headers=headers
        )
        
        # Handle None response from _make_request
        if response is None:
            raise RagMetricsAPIError(f"Failed to save {self.object_type}: No response received from server (check connection or base URL).")

        if response.status_code == 200:
            json_resp = response.json()
            # The API often returns the saved object nested under its type, e.g., {"dataset": {"id": ...}}
            saved_object_data = json_resp.get(self.object_type, {})
            new_id = saved_object_data.get("id")
            if new_id:
                self.id = new_id # type: ignore
            else:
                # Fallback if ID is directly at the root or if the key differs (less common)
                self.id = json_resp.get("id") # type: ignore
            return response # Return the full response object
        else:
            raise RagMetricsAPIError(f"Failed to save {self.object_type}", status_code=response.status_code, response_text=response.text)

    @classmethod
    def download(cls, id=None, name=None):
        """
        Download an object from the RagMetrics API.
        
        This method retrieves an object from the RagMetrics API by its ID or name.

    
        Args:
            id: ID of the object to download (mutually exclusive with name).
            name: Name of the object to download (mutually exclusive with id).

    
        Returns:
            RagMetricsObject: The downloaded object instance.
            
    
        Raises:
            ValueError: If neither id nor name is provided, or if object_type is not defined.
            Exception: If the API request fails.
        """
        from ragmetrics.api import ragmetrics_client, RagMetricsAuthError, RagMetricsAPIError # Import exceptions
        
        # Check auth *before* doing anything else
        if not ragmetrics_client.access_token:
            raise RagMetricsAuthError("RagMetrics client not authenticated. Please login first.")
        if not cls.object_type:
            raise ValueError("object_type must be defined for the class.")
        if id is None and name is None:
            raise ValueError("Either id or name must be provided for download.")
        
        params = {}
        if id is not None:
            params['id'] = id
        if name is not None:
            params['name'] = name
            
        endpoint = f"/api/client/{cls.object_type}/download/"
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        
        response = ragmetrics_client._make_request(
            method="get", endpoint=endpoint, params=params, headers=headers
        )
        
        # Handle None response from _make_request
        if response is None:
            raise RagMetricsAPIError(f"Failed to download {cls.object_type}: No response received from server (check connection or base URL).")

        if response.status_code == 200:
            json_resp = response.json()
            # The API often returns the object nested under its type, e.g., {"dataset": {"id": ..., "name": ...}}
            obj_data = json_resp.get(cls.object_type, None)
            if obj_data is None:
                # Fallback if the data is directly at the root (less common for current API)
                obj_data = json_resp 
            
            if not obj_data or not isinstance(obj_data, dict):
                 raise Exception(f"Failed to parse {cls.object_type} data from API response: {json_resp}")

            obj = cls.from_dict(obj_data)
            # Ensure the ID from the downloaded data is set on the instance
            downloaded_id = obj_data.get("id")
            if downloaded_id:
                obj.id = downloaded_id # type: ignore
            return obj
        else:
            raise RagMetricsAPIError(f"Failed to download {cls.object_type}", status_code=response.status_code, response_text=response.text) 