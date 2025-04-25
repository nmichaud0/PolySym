import subprocess
import requests
from deap import gp
import importlib.resources as pkg_resources
from ollama import chat, Client
import re

def round_ind(ind_str):

    float_pattern = re.compile(r'-?\d+\.\d+')

    def _round_match(m: re.Match) -> str:
        num = float(m.group())
        return f"{num:.2f}"

    return float_pattern.sub(_round_match, ind_str)

def strip_from_prefix(s: str) -> str:
    """
    Return s starting at the first occurrence of 'binary_' or 'unary_'.
    If neither is found, return s unchanged.
    """
    i_bin   = s.find("binary_")
    i_unary = s.find("unary_")
    # collect only non-negative indices
    candidates = [i for i in (i_bin, i_unary) if i != -1]
    if not candidates:
        return s
    # pick the earliest occurrence
    start = min(candidates)
    return s[start:]

"""class FunSearchConfigurator:
    def __init__(self, provider):"""


class FunSearchModel:
    def __init__(self,
                 pset: gp.PrimitiveSet,
                 provider: str = "ollama",
                 ollama_model: str = "llama2",
                 openrouter_api_key: str = None,
                 openrouter_model: str = "gpt-4o"):

        self.pset = pset
        self.provider = provider.lower()
        self.ollama_model = ollama_model
        self.openrouter_key = openrouter_api_key
        self.openrouter_model = openrouter_model

        self.base_prompt = pkg_resources.read_text(__package__, "prompt.txt")

        prim_names = [p.name for p in pset.primitives[pset.ret]]
        term_names = [i.name for i in list(pset.terminals.values())[0] if i.name.startswith('v') or i.name.startswith('x')]

        names = '\n'.join(prim_names) + '\n\nVariables:\n' + '\n'.join(term_names)
        self.base_prompt = self.base_prompt.replace('£££', names)

        if self.provider == "openrouter" and not self.openrouter_key:
            raise ValueError("OpenRouter provider requires an API key.")

        if self.provider == 'ollama':
            self.client = Client()
        else:
            self.client = None

    def _build_prompt(self,
                      pop: list[gp.PrimitiveTree],
                      fitnesses: list[float],
                      objective: str,
                      n_offspring: int = 10) -> str:

        lines = [f'Current population and fitnesses, the objective is {objective}:']

        for ind, fit in zip(pop, fitnesses):
            lines.append(f'- Expr: {round_ind(str(ind))}, Depth: {ind.height}, Fitness: {fit:.4f}')

        lines.append(f'\nGenerate {n_offspring} new valid expressions (prefix notation), one per line:')

        return self.base_prompt.replace('$$$', '\n'.join(lines))

    def _call_ollama(self, prompt: str, temperature: float) -> str:



        response = self.client.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}],
                                    options={'temperature': temperature}).message.content

        if '<think>' in response:
            response = response.split('</think>')[-1]

        return response


    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter HTTP API."""
        url = "https://openrouter.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.openrouter_model,
            "messages": [{"role":"user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def propose(self,
                pop: list[gp.PrimitiveTree],
                fitnesses: list[float],
                objective: str,
                n_offspring: int = 10,
                temperature=.1) -> list[gp.PrimitiveTree]:
        """
        Given a population and fitnesses, ask the LLM for new expressions and parse them.

        Returns:
            A list of valid DEAP PrimitiveTree individuals.
        """
        prompt = self._build_prompt(pop, fitnesses, objective, n_offspring)

        if self.provider == "ollama":
            raw = self._call_ollama(prompt, temperature=temperature)
        else:
            raw = self._call_openrouter(prompt)

        print('Got response')

        # split lines and parse
        candidates = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # optionally remove leading dashes or bullets
            if not line.startswith(('unary', 'binary')):
                line = strip_from_prefix(line)
            try:
                tree = gp.PrimitiveTree.from_string(line, self.pset)
                candidates.append(tree)
            except Exception:
                print(f'Failed to build tree with line {line}')
                continue

        return candidates, raw
