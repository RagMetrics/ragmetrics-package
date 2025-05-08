# test/test_utils.py
import pytest
from ragmetrics.utils import default_input, default_output, default_callback

# --- Tests for default_input --- 

def test_default_input_string():
    assert default_input("hello world") == "hello world"

def test_default_input_dict():
    assert default_input({"role": "user", "content": "find info"}) == "find info"

def test_default_input_dict_no_content():
    assert default_input({"role": "user"}) == str({"role": "user"})

def test_default_input_list_simple():
    assert default_input(["message 1", "message 2"]) == "message 2"

def test_default_input_list_dicts():
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "second turn"}
    ]
    assert default_input(messages) == "second turn"

def test_default_input_list_empty():
    assert default_input([]) == str([])

def test_default_input_object_with_content():
    class MockMessage:
        def __init__(self, content):
            self.content = content
    assert default_input(MockMessage("object content")) == "object content"

def test_default_input_object_without_content():
    class MockMessage:
        def __str__(self): return "MockStr"
    assert default_input(MockMessage()) == str(MockMessage())

def test_default_input_none():
     assert default_input(None) == ""

def test_default_input_empty_string():
     assert default_input("") == ""

def test_default_input_int():
    assert default_input(123) == "123"

# --- Tests for default_output --- 

def test_default_output_string():
    assert default_output("Simple output") == "Simple output"

def test_default_output_openai_dict_content():
    response = {"choices": [{"message": {"role": "assistant", "content": "AI response here"}}]}
    assert default_output(response) == "AI response here"

def test_default_output_openai_dict_tool_call_simple_args():
     response = {
         "choices": [{
             "message": {
                 "role": "assistant",
                 "content": None,
                 "tool_calls": [{
                     "id": "call_abc",
                     "type": "function",
                     "function": {"name": "search_docs", "arguments": '{\"query\": \"ragmetrics\"}'} 
                 }]
             }
         }]
     }
     assert default_output(response) == "=search_docs(query='ragmetrics')"

def test_default_output_openai_dict_tool_call_multiple_args():
     response = {
         "choices": [{
             "message": {
                 "role": "assistant",
                 "content": None,
                 "tool_calls": [{
                     "id": "call_def",
                     "type": "function",
                     "function": {"name": "calculate", "arguments": '{\"expr\": \"1+1\", \"precision\": 2}'} 
                 }]
             }
         }]
     }
     # Order might vary based on json loading, test both possibilities if needed, or sort keys
     expected1 = "=calculate(expr='1+1', precision=2)"
     expected2 = "=calculate(precision=2, expr='1+1')"
     actual = default_output(response)
     assert actual == expected1 or actual == expected2

def test_default_output_openai_obj_content():
    class MockChoice:
        class MockMessage:
            role = "assistant"
            content = "Object response"
            tool_calls = None
        message = MockMessage()
    class MockResponse:
        choices = [MockChoice()]
    assert default_output(MockResponse()) == "Object response"

def test_default_output_openai_obj_tool_call():
    class MockFunction:
        name = "get_weather"
        arguments = '{\"city\": \"London\"}'
    class MockToolCall:
        type = "function"
        function = MockFunction()
    class MockChoice:
        class MockMessage:
            role = "assistant"
            content = None
            tool_calls = [MockToolCall()]
        message = MockMessage()
    class MockResponse:
        choices = [MockChoice()]
    assert default_output(MockResponse()) == "=get_weather(city='London')"

def test_default_output_obj_with_text():
    class MockTextResponse:
        text = "Response via text attr"
    assert default_output(MockTextResponse()) == "Response via text attr"

def test_default_output_none():
    assert default_output(None) == ""

def test_default_output_empty_dict():
    assert default_output({}) == "{}"

def test_default_output_dict_with_content():
    assert default_output({"content": "Hello"}) == "Hello"

def test_default_output_dict_no_content():
    assert default_output({"role": "assistant"}) == str({"role": "assistant"})

# --- Tests for default_callback --- 

def test_default_callback_simple():
    result = default_callback("input query", "output answer")
    assert result == {"input": "input query", "output": "output answer"}

def test_default_callback_openai_like():
    in_msg = [{"role": "user", "content": "hello?"}]
    out_resp = {"choices": [{"message": {"role": "assistant", "content": "hi!"}}]}
    result = default_callback(in_msg, out_resp)
    assert result == {"input": "hello?", "output": "hi!"}

def test_default_callback_with_tool_call():
    in_msg = [{"role": "user", "content": "weather in sf"}]
    out_resp = {
         "choices": [{
             "message": {
                 "role": "assistant",
                 "content": None,
                 "tool_calls": [{
                     "id": "call_xyz",
                     "type": "function",
                     "function": {"name": "get_current_weather", "arguments": '{\"location\": \"San Francisco\", \"unit\": \"celsius\"}'} 
                 }]
             }
         }]
     }
    result = default_callback(in_msg, out_resp)
    expected_output1 = "=get_current_weather(location='San Francisco', unit='celsius')"
    expected_output2 = "=get_current_weather(unit='celsius', location='San Francisco')"
    assert result["input"] == "weather in sf"
    assert result["output"] == expected_output1 or result["output"] == expected_output2 