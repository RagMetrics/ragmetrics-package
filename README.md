<h1 align="center">
        <img src="https://cdn.simpleicons.org/python/fff/fff" alt="Python" width=24 height=24> 
Ragmetrics
    </h1>
    <p align="center">
        <p align="center">Call all LLM APIs using the OpenAI format.
        <br>
    </p>
<h4 align="center">
    <a href="https://pypi.org/project/ragmetrics-pkg/" target="_blank">
        <img src="https://img.shields.io/pypi/v/ragmetrics-pkg.svg" alt="PyPI Version">
    </a>
</h4>

Ragmetrics manages:

- Translate inputs to provider's `completion` endpoints
- Consistent `Traces` logged on [Ragmetrics portal](https://ragmetrics.ai/)
- Realtime Monitoring and AB Test Evaluations

Support for more providers. Contact our team to get it done.

# Usage Docs

```shell
pip install ragmetrics-pkg
```

#### Portal Login

```python
import ragmetrics

## login to ragmetrics account via portal key
ragmetrics.login(key="febfe*****************", off=False)
```

Here off flag helps developer to toggle traces.\
`off -> True` (will turn off the traces and vice versa)

#### Monitoring/AB Testing/Evaluation (based on compare models)

```python
import ragmetrics

## starts monitoring all LLM calls from this client
ragmetrics.monitor(client, context={"user": True})
```

Context will help to feed as dataset over Ragmetrics Tasks for Evaluation

### Code Utilisation

```python
from openai import OpenAI
import ragmetrics

client = OpenAI()

# Start monitoring all LLM calls from this client
ragmetrics.monitor(client, context={"user": True})

# Use regular OpenAI API calls with an extra optional metadata parameter
chat_completion = client.chat.completions.create(
  model="gpt-4o", 
  messages=[{"role": "user", "content": "Hello Ragmetrics"}],
  metadata={"pipelineStep":"generation", "property1":"Accuracy and Clarity"},
  comparison_model="gpt-4-turbo"
)
```

- Here `model` is to refer the response generation model `gpt-4o` in our case.
- `Message` with content for the actual developer's request to AI model.
- `Metadata` is to  drive the experiments as defined below:
  - `pipelineStep` for selecting the evaluation step as per Ragmetrics platform experiments.
  - `property` to add multiple criteria needs to be performed under evaluation (will be added with keys property1, 
    property2... etc.)

# Code Examples

#### Example 1

```python
from openai import OpenAI
import ragmetrics

ragmetrics.login(key="febfe*****************", off=False)

client = OpenAI()

ragmetrics.monitor(client, context={"user": True})

chat_completion = client.chat.completions.create(
  model="gpt-4o", 
  messages=[{"role": "user", "content": "Hello World !!!"}],
  metadata={"pipelineStep":"generation", "property1":"Accuracy and Clarity"},
  comparison_model="gpt-4-turbo"
  )
```

#### Example 2

```python
from openai import OpenAI
import ragmetrics

ragmetrics.login(key="febfe*****************", off=False)

client = OpenAI()

ragmetrics.monitor(client, context={"user": True})

chat_completion = client.chat.completions.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "How's the day today?"}],
  metadata={"pipelineStep":"generation", "property1":"Accuracy and Clarity", "property2":"Buddy-Friendly Tone"},
  comparison_model="gpt-4o"
  )
```

## Portal Keys UI on Ragmetrics Portal

![portal_keys.png](portal_keys.png)

# Why did we build this

- **Need for simplicity**: We need to provide developers a hassle-free solution to interact and monitor their 
  interaction with LLM models like Azure, OpenAI, LiteLLM ... etc.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/RagMetrics/ragmetrics-package/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=RagMetrics/ragmetrics-package" />
</a>
