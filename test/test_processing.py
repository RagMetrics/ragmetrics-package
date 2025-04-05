from ragmetrics.api import default_input, default_output

# Test with dictionary input
dict_input = {'role': 'user', 'content': 'Hello world'}
print('Dict input:', default_input(dict_input))

# Test with list input
list_input = [
    {'role': 'user', 'content': 'Hello'}, 
    {'role': 'assistant', 'content': 'Hi there'}
]
print('List input:', default_input(list_input))

# Test with object that has role and content attributes
class Message:
    def __init__(self):
        self.role = 'system'
        self.content = 'Test message'
        
obj_input = Message()
print('Object input:', default_input(obj_input))

# Test output with choices (mocking OpenAI response)
class Choice:
    def __init__(self):
        self.message = Message()
        
class Response:
    def __init__(self):
        self.choices = [Choice()]
        
openai_like = Response()
print('OpenAI-like output:', default_output(openai_like))

# Test output with text attribute
class TextResponse:
    def __init__(self):
        self.text = 'Response text'
        
text_resp = TextResponse()
print('Text output:', default_output(text_resp))

# Test output with content attribute
class ContentResponse:
    def __init__(self):
        self.content = 'Response content'
        
content_resp = ContentResponse()
print('Content output:', default_output(content_resp)) 