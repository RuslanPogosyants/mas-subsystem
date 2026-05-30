"""TestGeneratorAgent (F4): a Summary -> a quiz of QuizQuestions via an LlmAdapter.

One LLM call (retry once) asks for a JSON quiz; the response is validated as a
list of schemas.QuizQuestion (the Literal `type` enforces the supported question
types, extra="forbid" keeps it strict). An empty summary or an invalid/empty quiz
is refused; otherwise the agent informs {quiz_id, questions, difficulty}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents._llm_json import parse_with_retry
from src.agents.base import AgentBase
from src.core.schemas import Operation, QuizQuestion

if TYPE_CHECKING:
    from src.adapters.llm import LlmAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message

_MAX_PARSE_RETRIES: Final[int] = 1
_SYSTEM_PROMPT: Final[str] = (
    "Ты составляешь проверочный тест по учебному саммари. Ответь СТРОГО одним "
    'JSON-объектом {"questions": [...]}. Каждый вопрос: {question, type, choices, '
    "answer_idx, answer_indices}, где type — одно из "
    "single_choice, multi_choice, open_answer. Без markdown и пояснений."
)
_DEFAULT_NUM_QUESTIONS: Final[int] = 5
_DEFAULT_DIFFICULTY: Final[str] = "medium"


class _RawQuizQuestion(BaseModel):
    """The per-question shape the LLM emits.

    Lenient (extra="ignore") so stray fields the model invents — e.g. a numeric
    source_chunk_id, which real GigaChat fills with an ordinal — do not fail the
    whole quiz. The chunk linkage is an internal id the LLM cannot know; the agent
    leaves it None. The Literal type still enforces the supported question types.
    """

    model_config = ConfigDict(extra="ignore")

    question: str
    type: Literal["single_choice", "multi_choice", "open_answer"]
    choices: list[str] = Field(default_factory=list)
    answer_idx: int | None = None
    answer_indices: list[int] | None = None


class _RawQuiz(BaseModel):
    questions: list[_RawQuizQuestion]


class TestGeneratorAgent(AgentBase):
    name = "TestGeneratorAgent"

    def __init__(self, *, bus: RedisStreamBus, llm: LlmAdapter) -> None:
        super().__init__(
            bus=bus,
            channel="agent.test_generator",
            group="worker-test_generator",
            operation=Operation.F4_TEST,
        )
        self._llm = llm

    async def handle(self, message: Message) -> Message | None:
        summary_text = _summary_to_text(message.content.get("summary"))
        if not summary_text:
            return self._refuse(message, reason="no summary content for test generation")
        num_questions = message.content.get("num_questions", _DEFAULT_NUM_QUESTIONS)
        difficulty = message.content.get("difficulty", _DEFAULT_DIFFICULTY)
        if not isinstance(difficulty, str):
            difficulty = _DEFAULT_DIFFICULTY
        user = f"Составь {num_questions} вопрос(ов) сложности '{difficulty}' по саммари:\n{summary_text}"
        quiz = await parse_with_retry(
            self._llm, system=_SYSTEM_PROMPT, user=user, model_cls=_RawQuiz, retries=_MAX_PARSE_RETRIES
        )
        if quiz is None or not quiz.questions:
            return self._refuse(message, reason="llm returned invalid quiz json")
        questions = [QuizQuestion(**raw.model_dump()) for raw in quiz.questions]
        return self._inform(
            message,
            content={
                "quiz_id": f"quiz-{message.task_id}",
                "questions": [question.model_dump() for question in questions],
                "difficulty": difficulty,
            },
        )


def _summary_to_text(summary: object) -> str:
    if not isinstance(summary, dict):
        return ""
    sections = summary.get("sections")
    if not isinstance(sections, list):
        return ""
    parts = [str(section.get("text", "")) for section in sections if isinstance(section, dict)]
    return "\n".join(part for part in parts if part).strip()
