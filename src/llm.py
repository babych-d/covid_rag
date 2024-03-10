import os

import transformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from torch import cuda, bfloat16

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]


def get_llm():
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        token=HF_AUTH_TOKEN
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device,
        token=HF_AUTH_TOKEN,
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_AUTH_TOKEN,
    )

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        do_sample=False,
        max_new_tokens=512,
        repetition_penalty=1.1,
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm
