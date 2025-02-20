from dotenv import load_dotenv
load_dotenv(".env")

class StubLLMClient:
    def __init__(self, preset_input, preset_output):
        self.preset_input = preset_input
        self.preset_output = preset_output

    def chat(self):
        class Completions:
            def __init__(self, input_msg, output_msg):
                self.input_msg = input_msg
                self.output_msg = output_msg

            def create(self, messages, **kwargs):
                class Response:
                    def __init__(self, content):
                        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]
                return Response(self.output_msg)

        return type('Chat', (), {'completions': Completions(self.preset_input, self.preset_output)})()

# Initialize RagMetrics with API key from environment
import ragmetrics
ragmetrics.login(base_url="http://localhost:8000")

# Create and use the stub client
stub_client = StubLLMClient(
    preset_input=[{"role": "user", "content": "Test 123"}],
    preset_output="Response 123"
)

# Monitor the stub client
ragmetrics.monitor(stub_client, context={"test_case": "capital_cities"})

# The monitored client can now be used and will log traces
response = stub_client.chat.completions.create(
    messages=stub_client.preset_input,
    metadata={"source": "stub_test"}
)
