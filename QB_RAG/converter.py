from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field, RootModel
from QB_RAG.output_parser import get_json_format_instructions, OutputParser
from QB_RAG.prompt import Prompt, PromptValue
from QB_RAG.utils import ensembler
import typing as t


class QuestionList(BaseModel):
    questions: t.List[str] = Field(
        ..., description="list of questions from the context"
    )

    def dicts(self) -> dict:
        return self.model_dump()


_questions_output_instruction = get_json_format_instructions(QuestionList)
_statements_output_parser = OutputParser(pydantic_object=QuestionList)

QUESTION_GENERATION_PROMPT = Prompt(
    instruction="You are presented with a text authored by Large Language Model professionals, offering advice and strategies for task-specific evaluations. Your task is to formulate relevant questions that the text is written to address. Closely follow the example questions for style and structure when formulating your own questions for the provided text. Your generated questions should be in the first person with details, but only at a high school reading level. Your questions should be answerable from the text, but do not copy the text verbatim. MAKE SURE to generate at least the specified number of questions",
    output_format_instruction=_questions_output_instruction,
    examples=[
        {
            "context": """Abstractive summarization is the task of generating concise summaries that capture the key ideas in a source document. Unlike extractive summarization, which lifts entire sentences from the original text, abstractive summarization involves rephrasing and condensing information to create a newer, shorter version. This process requires a deep understanding of the content, the ability to identify the most important points, and a careful approach to avoid introducing hallucination defects. To evaluate abstractive summaries, Kryscinski et al. (2019) proposed four key dimensions: fluency, coherence, consistency, and relevance. Fluency asks whether sentences in the summary are well-formed and easy to read, while coherence examines whether the summary as a whole makes sense and is logically organized. Consistency checks whether the summary accurately reflects the content of the source document, ensuring no new or contradictory information is added. Lastly, relevance evaluates whether the summary focuses on the most important aspects of the source document, including key points and excluding less relevant details.""",
            "questions_generated": 7,
            "output": QuestionList.model_validate(
                {
                    "questions": [
                        "What is the main difference between abstractive and extractive summarization?",
                        "How does abstractive summarization condense information from a source document?",
                        "What are the four key dimensions proposed by Kryscinski et al. (2019) for evaluating abstractive summaries?",
                        "Why is it important for a summary to maintain fluency?",
                        "What does coherence refer to when evaluating a summary?",
                        "How does consistency in a summary ensure it accurately reflects the source document?",
                        "What aspects of a source document should a relevant summary focus on?",
                    ]
                }
            ).dicts(),
        }
    ],
    input_keys=["context", "questions_generated"],
    output_key="output",
    language="english",
)


class QuestionAnswerablity(BaseModel):
    question: str = Field(..., description="a question generated form the context")
    explanation: str = Field(..., description="the reason justifying")
    relevant: int = Field(
        ...,
        description="(0/1) if the content contains relevant information to infer an answer to the query",
    )


class QuestionsAnswerablity(RootModel):
    root: t.List[QuestionAnswerablity]

    def dicts(self) -> t.List[t.Dict]:
        return self.model_dump()


_answerablity_output_instructions = get_json_format_instructions(QuestionsAnswerablity)
_answerablity_output_parser = OutputParser(pydantic_object=QuestionsAnswerablity)


ANSWERABLITY_PROMPT = Prompt(
    instruction='Given a pair of user query and a paragraph of content, determine if the content contains relevant information to infer an answer to the query. Think step by step . First provide an explanation, then generate a 1 or 0 label. Put the results in a Python dictionary format with keys "explanation" and "relevant".',
    output_format_instruction=_answerablity_output_instructions,
    examples=[
        {
            "context": """Abstractive summarization is the task of generating concise summaries that capture the key ideas in a source document. Unlike extractive summarization, which lifts entire sentences from the original text, abstractive summarization involves rephrasing and condensing information to create a newer, shorter version. This process requires a deep understanding of the content, the ability to identify the most important points, and a careful approach to avoid introducing hallucination defects. To evaluate abstractive summaries, Kryscinski et al. (2019) proposed four key dimensions: fluency, coherence, consistency, and relevance. Fluency asks whether sentences in the summary are well-formed and easy to read, while coherence examines whether the summary as a whole makes sense and is logically organized. Consistency checks whether the summary accurately reflects the content of the source document, ensuring no new or contradictory information is added. Lastly, relevance evaluates whether the summary focuses on the most important aspects of the source document, including key points and excluding less relevant details.""",
            "questions": [
                "How does coherence affect the evaluation of a summary?",
                "What does consistency check in an abstractive summary?",
                "Why can similarity-based evaluations like BERTScore and MoverScore be unreliable?",
                "Why can metrics like ROUGE and METEOR lead to high variance from the ground truth?",
            ],
            "output": QuestionsAnswerablity.model_validate(
                [
                    {
                        "question": "How does coherence affect the evaluation of a summary?",
                        "explanation": "Coherence examines whether the summary as a whole makes sense and is logically organized.",
                        "relevant": 1,
                    },
                    {
                        "question": "What does consistency check in an abstractive summary?",
                        "explanation": "Consistency checks whether the summary accurately reflects the content of the source document, ensuring no new or contradictory information is added.",
                        "relevant": 1,
                    },
                    {
                        "question": "Why can similarity-based evaluations like BERTScore and MoverScore be unreliable?",
                        "explanation": "The context does not address the reliability or issues related to similarity-based evaluations such as BERTScore or MoverScore.",
                        "relevant": 0,
                    },
                    {
                        "question": "Why can metrics like ROUGE and METEOR lead to high variance from the ground truth?",
                        "explanation": "The context does not discuss ROUGE, METEOR, or their variance from the ground truth.",
                        "relevant": 0,
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["context", "questions"],
    output_key="output",
    language="english",
)


class Converter:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLM,
        questions_generated: int = 10,
        max_retries: int = 1,
        reproducibility: int = 1,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.max_retries = max_retries
        self._reproducibility = reproducibility
        self.questions_generated = questions_generated

    def _create_question_generation_prompt(self, document: str) -> PromptValue:
        context = document
        # you can add a custom logic on how many questions you would
        # like to generate based the length/complexity of context
        questions_generated = self.questions_generated

        prompt_value = QUESTION_GENERATION_PROMPT.format(
            **{"context": context, "questions_generated": questions_generated}
        )
        return prompt_value

    def _create_answerablity_prompt(
        self, context: str, questions: t.List[str]
    ) -> PromptValue:
        prompt_value = ANSWERABLITY_PROMPT.format(
            **{"context": context, "questions": questions}
        )
        return prompt_value

    def add_documents(self, document: str):
        assert self.llm is not None, "LLM is not set"
        que_gen_prompt = self._create_question_generation_prompt(document)
        questions = self.llm.generate([que_gen_prompt])

        questions = _statements_output_parser.parse(
            questions.generations[0][0].text, que_gen_prompt, self.llm, self.max_retries
        )

        questions = [que for que in questions.dicts()["questions"]]

        assert isinstance(questions, t.List), "questions must be a list"

        ans_prompt = self._create_answerablity_prompt(document, questions)
        answerablity_result = self.llm.generate(
            [ans_prompt],
            n=self._reproducibility,
        )

        answerablity_result = [
            answerablity_result.generations[0][i].text
            for i in range(self._reproducibility)
        ]

        answerablity_list = [
            _answerablity_output_parser.parse(text, ans_prompt, self.llm, max_retries=1)
            for text in answerablity_result
        ]

        answerablity_list = [
            que.dicts() for que in answerablity_list if que is not None
        ]

        if answerablity_list:
            answerablity_list = ensembler.from_discrete(
                answerablity_list,
                "relevant",
            )

            answerablity_list = QuestionsAnswerablity.model_validate(answerablity_list)

        document_list = []

        for q in answerablity_list.root:
            if q.relevant == 1:
                document_list.append(
                    Document(page_content=q.question, metadata={"context": document})
                )

        self.vector_store.add_documents(document_list)
        return len(document_list)
