from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

SECRET_PATH = '../secret.txt'
# Determines level of creativity, where 0 is very deterministic (always chooses most probable word) and 2 is very creative
TEMPERATURE_VALUE = 0.3
# Determine maximum number of new tokens that the model is allowed to generate
MAX_NEW_TOKENS = 500


class LlmModel:
    @staticmethod
    def wrap_prompt(prompt):
        return f"""<s>[INST]{prompt} \n Know that I trust you and believe in your capabilities. Take your time and think step by step.[/INST]</s>"""

    @staticmethod
    def get_prompt_to_personal_url(urls: str):
        return LlmModel.wrap_prompt(f"""You are a helpful assistant that need to help me decide whether a give URL is personal or not.
        The URL is personal if it has some kind of information that might lead to a real human.
        You will be given a list of URLs and you should return for each one of them if they are personal or not.
        Here are some examples of a personal urls:
        1. https://www.johndoeportfolio.com
        2. http://www.sarah-blog.net
        3. ttps://www.linkedin.com/in/jane-smith
        4. google.com
        5. twitter.com/alex_thompson
        6. facebook.com/marcom
        The expected answer is:
        1. Yes
        2. Yes
        3. Yes
        4. No
        5. Yes
        6. Yes
        You should answer only "yes" or "no".
        It is imperative to do not generate any more text besides those two words!
        Ensure that you answer to the exact number of urls. such that if I provided you 200 urls, you should return 200 yes/no answers
        Now it is real, here are the URLs: {urls}""")

    def __init__(self, token_path=SECRET_PATH, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        self.token_path = token_path
        self.model_id = model_id
        self.token = self.read_token()
        self.client = self.get_client()

    def read_token(self):
        with open(self.token_path, 'r') as file:
            token = file.read()
        return token

    def get_client(self):
        return InferenceClient(model=self.model_id, token=self.token)

    def query_llm(self, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature_value=TEMPERATURE_VALUE, seed=0):
        return self.client.text_generation(prompt=prompt,
                                           max_new_tokens=max_new_tokens,
                                           temperature=temperature_value,
                                           seed=seed)

    def query_ner(self, text):
        return self.nlp(text)
