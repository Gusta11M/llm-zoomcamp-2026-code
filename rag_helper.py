

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        google_types,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gemini-3.1-flash-lite",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.google_types = google_types

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, user_prompt):

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config=self.google_types.GenerateContentConfig(
                system_instruction=self.instructions
            )
        )

        return response.text
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

class RAGDocs:

    def __init__(
        self,
        index,
        llm_client,
        google_types,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        filename_filter=None,
        model="gemini-3.1-flash-lite",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.filename_filter = filename_filter
        self.model = model
        self.google_types = google_types

    def search(self, query, num_results=5):
        filter_dict = {}
        if self.filename_filter:
            filter_dict["filename"] = self.filename_filter

        return self.index.search(
            query,
            num_results=num_results,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"# Source: {doc['filename']}")
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, user_prompt):
        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config=self.google_types.GenerateContentConfig(
                system_instruction=self.instructions
            ),
        )
        self.last_usage = response.usage_metadata
        return response.text                        

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer



