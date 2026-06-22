INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        # filename='llm-zoomcamp',
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        # self.filename = filename
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # filter_dict = {'filename': self.filename}

        return self.index.search(
            query,
            num_results=num_results,
            # filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        # return response.output_text
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        text_response = answer.output_text.strip()
        usage = answer.usage

        usage_dict = None
        if usage is not None:
            usage_dict = {
                'input_tokens': getattr(usage, 'input_tokens', None),
                'output_tokens': getattr(usage, 'output_tokens', None),
                'total_tokens': getattr(usage, 'total_tokens', None),
            }

        return {
            'text': text_response,
            'usage': usage_dict,
        }
