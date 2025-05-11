"""
OpenAI v0.x (legacy) API integration for RagMetrics.
This module provides wrapper functions for the module-style OpenAI API used in v0.28.x and earlier.
"""

import time
import types
import logging
import functools
from typing import Any, Optional, Dict, Callable

logger = logging.getLogger(__name__)

def is_openai_v0():
    """Check if we're using OpenAI v0.x"""
    try:
        import openai
        openai_version = getattr(openai, '__version__', '0.0.0')
        return not openai_version.startswith('1.')
    except ImportError:
        return False

def wrap_openai_v0_chat_completion(rm_client: Any, openai_module: Any, callback: Optional[Callable] = None) -> bool:
    """
    Wrap the OpenAI v0.x ChatCompletion.create method for monitoring.
    
    Args:
        rm_client: RagMetricsClient instance
        openai_module: The openai module (not a class instance)
        callback: Optional callback function for custom processing
        
    Returns:
        bool: True if wrapping was successful
    """
    if not is_openai_v0():
        logger.debug("OpenAI v0.x wrapper skipped - not running on v0.x")
        return False
        
    if not hasattr(openai_module, 'ChatCompletion') or not hasattr(openai_module.ChatCompletion, 'create'):
        logger.warning("OpenAI module does not have ChatCompletion.create method")
        return False
    
    # Store the original method
    original_create = openai_module.ChatCompletion.create
    
    # Create wrapper function
    def wrapped_create(*args, **kwargs):
        # Extract contextual data
        contexts = kwargs.pop("contexts", None)
        expected = kwargs.pop("expected", None)
        user_metadata = kwargs.pop("metadata", {})
        force_new_conversation = kwargs.pop("force_new_conversation", False)
        
        # Get model and other info
        model_name = kwargs.get("model", None)
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", None)
        tool_choice = kwargs.get("tool_choice", None)
        
        # Only use the metadata provided by the user, don't add OpenAI parameters
        # We want to preserve user's metadata without adding our own
        metadata_llm = user_metadata.copy() if user_metadata else {}
        
        # Measure time
        start_time = time.time()
        
        # Call the API
        try:
            response = original_create(*args, **kwargs)
            error = None
        except Exception as e:
            error = e
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Don't log if error and no response
            if error is not None and 'response' not in locals():
                return
            
            # Process callback if provided
            callback_result = None
            if callback and 'response' in locals():
                try:
                    callback_result = callback(messages, response)
                except Exception as e:
                    logger.error(f"Callback error in OpenAI v0.x wrapper: {e}")
            
            # Log trace with RagMetrics
            rm_client._log_trace(
                input_messages=messages,
                response=response if 'response' in locals() else None,
                metadata_llm=metadata_llm,
                contexts=contexts,
                expected=expected,
                duration=duration,
                model_name=model_name,
                tools=tools,
                tool_choice=tool_choice,
                callback_result=callback_result,
                force_new_conversation=force_new_conversation,
                error=error
            )
        
        return response
    
    # Replace the original method
    openai_module.ChatCompletion.create = wrapped_create
    logger.info("Successfully wrapped OpenAI v0.x ChatCompletion.create method")
    return True

def wrap_openai_v0_completion(rm_client: Any, openai_module: Any, callback: Optional[Callable] = None) -> bool:
    """
    Wrap the OpenAI v0.x Completion.create method for monitoring.
    Similar to the ChatCompletion wrapper but for regular completions.
    """
    if not is_openai_v0():
        logger.debug("OpenAI v0.x wrapper skipped - not running on v0.x")
        return False
        
    if not hasattr(openai_module, 'Completion') or not hasattr(openai_module.Completion, 'create'):
        logger.warning("OpenAI module does not have Completion.create method")
        return False
    
    # Store the original method
    original_create = openai_module.Completion.create
    
    # Create wrapper function
    def wrapped_create(*args, **kwargs):
        # Extract contextual data
        contexts = kwargs.pop("contexts", None)
        expected = kwargs.pop("expected", None)
        user_metadata = kwargs.pop("metadata", {})
        force_new_conversation = kwargs.pop("force_new_conversation", False)
        
        # Get model and other info
        model_name = kwargs.get("model", None)
        prompt = kwargs.get("prompt", "")
        
        # Only use the metadata provided by the user, don't add OpenAI parameters
        # We want to preserve user's metadata without adding our own
        metadata_llm = user_metadata.copy() if user_metadata else {}
        
        # Measure time
        start_time = time.time()
        
        # Call the API
        try:
            response = original_create(*args, **kwargs)
            error = None
        except Exception as e:
            error = e
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Don't log if error and no response
            if error is not None and 'response' not in locals():
                return
            
            # Process callback if provided
            callback_result = None
            if callback and 'response' in locals():
                try:
                    callback_result = callback(prompt, response)
                except Exception as e:
                    logger.error(f"Callback error in OpenAI v0.x wrapper: {e}")
            
            # Log trace with RagMetrics
            rm_client._log_trace(
                input_messages=prompt,
                response=response if 'response' in locals() else None,
                metadata_llm=metadata_llm,
                contexts=contexts,
                expected=expected,
                duration=duration,
                model_name=model_name,
                callback_result=callback_result,
                force_new_conversation=force_new_conversation,
                error=error
            )
        
        return response
    
    # Replace the original method
    openai_module.Completion.create = wrapped_create
    logger.info("Successfully wrapped OpenAI v0.x Completion.create method")
    return True 