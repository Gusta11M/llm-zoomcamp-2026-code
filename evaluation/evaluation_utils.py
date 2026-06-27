import time
import sys
sys.path.insert(0, '..')

from google.genai import types
from tqdm.auto import tqdm
from agentic_rag.rag_helper import RAGBase


def calc_price(usage):
    input_price_per_million = 0.075   # gemini-2.5-flash
    output_price_per_million = 0.30

    input_cost = (usage.prompt_token_count / 1_000_000) * input_price_per_million
    output_cost = (usage.candidates_token_count / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": f"{input_cost:.10f}",
        "output_cost": f"{output_cost:.10f}",
        "total_cost": f"{total_cost:.10f}",
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="gemini-2.5-flash"):
    config = types.GenerateContentConfig(
        system_instruction=instructions,
        response_mime_type="application/json",
        response_schema=output_type
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_prompt)]
        )
    ]

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

    return response.parsed, response.usage_metadata


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        config = types.GenerateContentConfig(
            system_instruction=self.instructions
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )

        self.last_usage = response.usage_metadata
        self.usages.append(response.usage_metadata)

        return response.text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results