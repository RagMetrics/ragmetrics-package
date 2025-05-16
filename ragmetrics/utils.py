import importlib
from typing import Dict, Any

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


def format_function_signature(func_name: str, args_dict: Dict[str, Any]) -> str:
    """
    Format a function signature into a string.
    
    Args:
        func_name: The name of the function.
        args_dict: Dictionary of argument names and values.
    
    Returns:
        str: Formatted function signature string in the format =function(arg1=val1, arg2=val2, ...)
    """
    # Format args as key=value pairs with proper quoting for strings
    args_str = ", ".join(
        f"{k}={repr(v) if isinstance(v, str) else v}" 
        for k, v in args_dict.items()
    )
    return f"={func_name}({args_str})"        