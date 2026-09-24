"""Thin Python client for a TokenVector.Inference server.

No native extension: it speaks the OpenAI-compatible HTTP API over
urllib, so any Python (>=3.8, stdlib only) can drive `tv/tools/cli.tv serve`.
This is the honest form a "binding" takes without a C ABI in the language.
"""
import json
import urllib.request

class TVClient:
    def __init__(self, base_url="http://127.0.0.1:8080", api_key=""):
        self.base = base_url.rstrip("/")
        self.api_key = api_key

    def _post(self, path, payload, stream=False):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.api_key} if self.api_key else {})})
        with urllib.request.urlopen(req) as r:
            body = r.read().decode()
            return body if stream else json.loads(body)

    def health(self):
        with urllib.request.urlopen(self.base + "/health") as r:
            return json.loads(r.read().decode())

    def chat(self, messages, stream=False, **kw):
        payload = {"messages": messages, "stream": stream}
        payload.update(kw)
        return self._post("/v1/chat/completions", payload, stream=stream)

    def complete(self, prompt, **kw):
        payload = {"prompt": prompt}
        payload.update(kw)
        return self._post("/v1/completions", payload)

    def embed(self, inputs):
        return self._post("/v1/embeddings", {"input": inputs})

    def rerank(self, query, documents, top_n=0):
        return self._post("/v1/rerank",
                          {"query": query, "documents": documents, "top_n": top_n})

    def predict(self, values):
        return self._post("/predict", values)
