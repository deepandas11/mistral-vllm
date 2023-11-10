import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download
HF_KEY = os.environ.get("HF_KEY")

class InferlessPythonModel:
    def initialize(self):
        self.template = """SYSTEM: You are a helpful assistant.
        USER: {}
        ASSISTANT: """
        snapshot_download(
            "mistralai/Mistral-7B-Instruct-v0.1",
            local_dir="/model",
            token=HF_KEY,
        )
        self.llm = LLM("/model")
    
    def infer(self, inputs):
        print("inputs[questions] -->", inputs["questions"], flush=True)
        prompts = [self.template.format(inputs["questions"])]
        print("Prompts -->", prompts, flush=True)
        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass
