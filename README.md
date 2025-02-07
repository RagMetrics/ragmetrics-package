<h1 align="center">RagMetrics</h1>
    <p align="center">
        Monitor your LLM calls. Test your LLM app.
    </p>
<h4 align="center">
    <a href="https://pypi.org/project/ragmetrics-pkg/" target="_blank">
        <img src="https://img.shields.io/pypi/v/ragmetrics-pkg.svg" alt="PyPI Version">
    </a>
</h4>

[RagMetrics](https://ragmetrics.ai/) offers:

- The best LLM judge on the market (>95% human agreement)
- A/B testing for your entire LLM pipeline
- Evaluations for retrievals, not just generation

With this package, you can log your LLM calls to RagMetrics and use them as labeled data

# Quickstart

```shell
pip install ragmetrics-pkg
```

```python
import ragmetrics
from openai import OpenAI

## login to ragmetrics account via portal key
ragmetrics.login(key="your_ragmetrics_key")
openai_client = OpenAI()

## Monitor OpenAI, LangChain or LiteLLM clients
ragmetrics.monitor(openai_client)

## Then use as usual. All calls will be logged
resp = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"What is the capital of Spain?"}]
)
```
## Get your key at RagMetrics.ai
