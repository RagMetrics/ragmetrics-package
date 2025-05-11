import json
import logging

logger = logging.getLogger(__name__)

def default_input(raw_input):
    """
    Format input messages into a standardized string format.
    
    Handles various input formats (list of messages, single message object, etc.)
    and converts them into a consistent string representation.
    
    Args:
        raw_input: The input to format. Can be a list of messages, a dictionary
                  with role/content keys, an object with role/content attributes,
                  or a primitive value.
    
    Returns:
        str: Formatted string representation of the input, or None if input is empty.
    """
    # Input processing
    if not raw_input:
        return None
        
    if isinstance(raw_input, list) and len(raw_input) > 0:
        raw_input = raw_input[-1]
    
    if isinstance(raw_input, dict) and "content" in raw_input:
        content = raw_input.get('content', '')
    elif hasattr(raw_input, "content"):
        content = raw_input.content
    else:
        content = str(raw_input)
    return content

def default_output(raw_response):
    """
    Extract content from various types of LLM responses.
    
    Handles different response formats from various LLM providers and APIs,
    extracting the actual content in a consistent way.
    
    Args:
        raw_response: The response object from the LLM. Can be OpenAI ChatCompletion,
                     object with text/content attributes, or another response format.
    
    Returns:
        str: The extracted content from the response, or the raw response if content
             cannot be extracted.
    """
    if not raw_response:
        return None
        
    # Handle tool_calls in the response (OpenAI function calling API)
    if isinstance(raw_response, dict) and "choices" in raw_response:
        try:
            message = raw_response["choices"][0]["message"]
            if message.get("tool_calls") and not message.get("content"):
                tool_call = message["tool_calls"][0]
                if tool_call["type"] == "function":
                    func_name = tool_call["function"]["name"]
                    # Parse the JSON arguments
                    args_dict = json.loads(tool_call["function"]["arguments"])
                    # Format args as key=value pairs with proper quoting for strings
                    args_str = ", ".join(
                        f"{k}={repr(v) if isinstance(v, str) else v}" 
                        for k, v in args_dict.items()
                    )
                    return f"={func_name}({args_str})"
        except Exception as e:
            logger.error("Error formatting tool_calls from response: %s", e)
            
    # Also handle object-style responses (OpenAI client library)
    if hasattr(raw_response, "choices") and raw_response.choices:
        try:
            message = raw_response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls and (not message.content or message.content is None):
                tool_call = message.tool_calls[0]
                if tool_call.type == "function":
                    func_name = tool_call.function.name
                    # Parse the JSON arguments
                    args_dict = json.loads(tool_call.function.arguments)
                    # Format args as key=value pairs with proper quoting for strings
                    args_str = ", ".join(
                        f"{k}={repr(v) if isinstance(v, str) else v}" 
                        for k, v in args_dict.items()
                    )
                    return f"={func_name}({args_str})"
            return message.content
        except Exception as e:
            logger.error("Error extracting content from response.choices: %s", e)
    
    # Handle other response types
    if hasattr(raw_response, "text"):
        content = raw_response.text
    elif hasattr(raw_response, "content"):
        content = raw_response.content
    else:
        content = str(raw_response)
    return content

def default_callback(raw_input, raw_output) -> dict:
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