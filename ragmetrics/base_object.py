import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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
        return cls(**data)

    def save(self):
        """
        Save the object to the RagMetrics API.
        
        This method sends the object to the RagMetrics API for storage and
        retrieves the assigned ID. Different object types may use different
        endpoints based on their needs.

        Returns:
            Response: The API response from saving the object. (Note: _make_request returns parsed data or raises)
            
        Raises:
            ValueError: If object_type is not defined.
            RagMetricsAPIError: If the API request fails.
            RagMetricsAuthError: If authentication fails.
            RagMetricsError: For other client-side issues (e.g. serialization).
        """
        # Import here to avoid circular imports
        from .api import ragmetrics_client, RagMetricsError, RagMetricsAPIError, RagMetricsAuthError
        
        if not self.object_type:
            raise ValueError("object_type must be defined.")
        
        try:
            payload = self.to_dict()
        except Exception as e:
            logger.error(f"Error serializing {self.object_type} for save: {e}", exc_info=True)
            raise RagMetricsError(f"Error during serialization for save: {e}") from e

        # Determine the appropriate endpoint based on object type and state
        if self.object_type == "trace" and not getattr(self, "edit_mode", False):
            # For new traces, use the logtrace endpoint
            endpoint = "/api/client/logtrace/"
            if 'raw' not in payload:
                payload = {"raw": payload}
        else:
            # For all other objects and for editing traces, use the standard save endpoint
            endpoint = f"/api/client/{self.object_type}/save/"
        
        try:
            response_data = ragmetrics_client._make_request(
                method="post", 
                endpoint=endpoint, 
                json=payload
            )

            if isinstance(response_data, dict):
                new_id = None
                # Handle different response formats
                if "id" in response_data:
                    new_id = response_data.get("id")
                elif self.object_type in response_data and isinstance(response_data[self.object_type], dict):
                    new_id = response_data[self.object_type].get("id")
                elif "trace" in response_data and isinstance(response_data["trace"], dict):
                    # For logtrace endpoint responses
                    new_id = response_data["trace"].get("id")
                
                if new_id:
                    self.id = new_id
                    logger.info(f"{self.object_type} saved successfully with ID: {self.id}")
                    return self 
                else:
                    logger.error(f"Saved {self.object_type} but no ID was found in response: {response_data}")
                    raise RagMetricsAPIError(f"Save successful for {self.object_type} but no ID returned.", response_text=str(response_data))
            else:
                logger.error(f"Unexpected response type from _make_request during save: {type(response_data)}. Response: {str(response_data)[:200]}")
                raise RagMetricsAPIError(f"Save failed for {self.object_type}: Unexpected response structure.", response_text=str(response_data))
        
        except (RagMetricsAPIError, RagMetricsAuthError, RagMetricsError) as e:
            logger.error(f"Failed to save {self.object_type}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during save of {self.object_type}: {e}")
            raise RagMetricsError(f"An unexpected error occurred while saving {self.object_type}: {e}") from e

    @classmethod
    def download(cls, id=None, name=None):
        """
        Download an object from the RagMetrics API by ID or name using the /api/client endpoint.
        Uses ragmetrics_client._make_request with Bearer token authentication.
        
        Args:
            id: The ID of the object to download
            name: The name of the object to download (if ID not provided)
            
        Returns:
            An instance of the class populated with data from the API
            
        Raises:
            RagMetricsAuthError: If not authenticated
            RagMetricsAPIError: If API request fails
            ValueError: If neither id nor name is provided
        """
        # Import here to avoid circular imports
        from .api import ragmetrics_client, RagMetricsError, RagMetricsAPIError, RagMetricsAuthError
        
        if not ragmetrics_client.access_token:
            raise RagMetricsAuthError("RagMetrics client not authenticated. Please login first.")
        if not cls.object_type:
            raise ValueError(f"object_type must be defined for class {cls.__name__} to download.")
        if id is None and name is None:
            raise ValueError("Either id or name must be provided for download.")
        if id is not None and name is not None:
             logger.warning("Both id and name provided to download. Using id.")
             name = None # Prioritize ID if both are given

        # Construct endpoint and query parameters
        endpoint = f"/api/client/{cls.object_type}/download/"
        params = {}
        if id is not None:
            params['id'] = id
        else: # name must be not None here
            params['name'] = name
        
        try:
            # Use _make_request (which handles Bearer token)
            response_data = ragmetrics_client._make_request(
                method="get", endpoint=endpoint, params=params
            )

            if isinstance(response_data, dict):
                # Get object data from response
                obj_data = response_data.get(cls.object_type)
                if not isinstance(obj_data, dict):
                    if response_data.get("status") == "success":
                        obj_data = response_data.get("trace")
                    else:
                        obj_data = response_data
                
                if not obj_data: # Handle empty response or unexpected structure
                    logger.error(f"No valid '{cls.object_type}' data found in download response for id={id}/name={name}. Response: {response_data}")
                    return None
                
                logger.debug(f"Creating {cls.__name__} from data: {obj_data}")
                obj = cls.from_dict(obj_data)
                # Ensure ID is set from the response data
                downloaded_id = obj_data.get("id")
                if downloaded_id:
                    obj.id = downloaded_id
                elif id is not None: # Fallback to the requested ID if not in response
                     obj.id = id
                return obj
            else:
                logger.error(f"Failed to get valid dict data for {cls.object_type} id={id}/name={name}. Response type: {type(response_data)}, Response: {str(response_data)[:200]}...")
                return None
        except RagMetricsAPIError as e:
            logger.error(f"API error while downloading {cls.object_type} id={id}/name={name}: {e}")
            raise # Re-raise API errors for clarity
        except RagMetricsAuthError as e:
            logger.error(f"Auth error while downloading {cls.object_type} id={id}/name={name}: {e}")
            raise # Re-raise auth errors
        except Exception as e:
            # Catch unexpected errors during download/parsing
            logger.exception(f"Unexpected error during download of {cls.object_type} id={id}/name={name}: {e}")
            raise RagMetricsError(f"Unexpected error during download: {e}") from e 