import importlib
import sys
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

def import_function(function):
    """
    Import a function from a string path or return the callable
    
    Args:
        function (str or callable or None):
            - String in the format "module.submodule.function_name" for imported functions
            - Simple string with just a function name (will be returned as-is for later handling)
            - Callable function
            - None
        
    Returns:
        callable or None or str: 
            - The imported function if successfully imported
            - None if input is None
            - The original string if it's just a function name without module path
        
    Raises:
        ValueError: If the function cannot be imported or is not callable
    """
    
    if function is None:
        return None
    elif callable(function):
        return function
    
    # If the function is a simple name without dots, just return it as-is
    # This allows the caller to handle simple function names differently
    if isinstance(function, str) and '.' not in function:
        return function
    
    try:
        # Split the path into module path and function name
        parts = function.split('.')
        if len(parts) < 2:
            raise ValueError(f"Function path '{function}' must be in the format 'module.function_name'")
            
        module_path = '.'.join(parts[:-1])
        function_name = parts[-1]
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the function
        imported_function = getattr(module, function_name)
        
        # Verify it's callable
        if not callable(imported_function):
            raise ValueError(f"Imported object '{function}' is not callable")
            
        return imported_function
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import function '{function}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Error importing function '{function}': {str(e)}")

def default_input(raw_input: Any) -> str:
    """
    Format input messages into a standardized string format.
    
    Handles various input formats (list of messages, single message object, etc.)
    and converts them into a consistent string representation.
    
    Args:
        raw_input: The input to format. Can be a list of messages, a dictionary
                  with role/content keys, an object with role/content attributes,
                  or a primitive value.
    
    Returns:
        str: Formatted string representation of the input, or an empty string if input is empty.
    """
    content_str = ""
    if isinstance(raw_input, list) and len(raw_input) > 0:
        # Process the last message if it's a list
        # This aligns with how it might be used for chat histories
        raw_input = raw_input[-1] 

    if isinstance(raw_input, dict) and "content" in raw_input:
        content = raw_input.get('content')
        content_str = str(content) if content is not None else ""
    elif hasattr(raw_input, "content"):
        content = raw_input.content
        content_str = str(content) if content is not None else ""
    elif raw_input is not None:
        content_str = str(raw_input)
    
    return content_str

def default_output(raw_response: Any) -> str:
    """
    Extract content from various types of LLM responses.
    
    Handles different response formats from various LLM providers and APIs,
    extracting the actual content in a consistent way.
    
    Args:
        raw_response: The response object from the LLM. Can be OpenAI ChatCompletion,
                     object with text/content attributes, or another response format.
    
    Returns:
        str: The extracted content from the response, or an empty string if content
             cannot be extracted or raw_response is None.
    """
    if raw_response is None:
        return ""
        
    content_str = ""

    # Handle tool_calls in the response (OpenAI function calling API - dict style)
    if isinstance(raw_response, dict) and "choices" in raw_response:
        try:
            message = raw_response["choices"][0]["message"]
            if message.get("tool_calls") and not message.get("content"):
                tool_call = message["tool_calls"][0]
                if tool_call["type"] == "function":
                    func_name = tool_call["function"]["name"]
                    args_dict = json.loads(tool_call["function"]["arguments"])
                    args_str = ", ".join(
                        f"{k}={repr(v) if isinstance(v, str) else v}" 
                        for k, v in args_dict.items()
                    )
                    return f"={func_name}({args_str})"
            elif message.get("content") is not None:
                 content_str = str(message.get("content"))
        except Exception as e:
            logger.error("Error formatting tool_calls from dict response: %s", e)
            
    # Also handle object-style responses (OpenAI client library, Anthropic, etc.)
    elif hasattr(raw_response, "choices") and raw_response.choices:
        try:
            message = raw_response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls and (not message.content or message.content is None):
                tool_call = message.tool_calls[0]
                if tool_call.type == "function":
                    func_name = tool_call.function.name
                    args_dict = json.loads(tool_call.function.arguments)
                    args_str = ", ".join(
                        f"{k}={repr(v) if isinstance(v, str) else v}" 
                        for k, v in args_dict.items()
                    )
                    return f"={func_name}({args_str})"
            elif hasattr(message, "content") and message.content is not None:
                 content_str = str(message.content)
        except Exception as e:
            logger.error("Error extracting content from response.choices: %s", e)

    # Fallbacks for other common response structures
    elif hasattr(raw_response, "text"): # e.g. Cohere
        content = raw_response.text
        content_str = str(content) if content is not None else ""
    elif hasattr(raw_response, "content") and not callable(raw_response.content): # Check if content is not a method
        # Some objects might have a 'content' attribute that is a list (e.g. Anthropic messages)
        if isinstance(raw_response.content, list):
            # Try to extract text from the first content block if it's a structured message
            if raw_response.content and hasattr(raw_response.content[0], 'text'):
                content = raw_response.content[0].text
                content_str = str(content) if content is not None else ""
            else: # Otherwise, convert the list to string
                content_str = str(raw_response.content)
        else:
            content = raw_response.content
            content_str = str(content) if content is not None else ""
    elif isinstance(raw_response, dict) and "content" in raw_response: # Added case for dict with 'content' key
        content = raw_response.get("content")
        content_str = str(content) if content is not None else ""
    
    # If content_str is still empty, try converting the whole raw_response to string
    if not content_str and raw_response is not None:
        content_str = str(raw_response)
        
    return content_str

def default_callback(raw_input: Any, raw_output: Any) -> dict:
    """
    Create a standardized callback result dictionary.
    
    This is the default callback used by the monitor function when no custom
    callback is provided.

    Args:
        raw_input: The raw input to the LLM.
        raw_output: The raw output from the LLM.

    Returns:
        dict: A dictionary containing formatted input and output.
    """
    return {
        "input": default_input(raw_input),
        "output": default_output(raw_output)
    } 