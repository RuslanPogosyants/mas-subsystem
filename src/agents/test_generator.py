"""TestGeneratorAgent (F4): a Summary -> a quiz of QuizQuestions via an LlmAdapter.

One LLM call (retry once) asks for a JSON quiz; the response is validated as a
list of schemas.QuizQuestion (the Literal `type` enforces the supported question
types, extra="forbid" keeps it strict). An empty summary or an invalid/empty quiz
is refused; otherwise the agent informs {quiz_id, questions, difficulty}.
"""

from __future__ import annotations

import re
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
_MIN_CONTENT_LEN: Final[int] = 4
_SYSTEM_PROMPT: Final[str] = (
    "Ты составляешь проверочный тест по учебному саммари. Ответь СТРОГО одним "
    'JSON-объектом {"questions": [...]}. Каждый вопрос: {question, type, choices, '
    "answer_idx, answer_indices}, где type — одно из "
    "single_choice, multi_choice, open_answer. Без markdown и пояснений.\n"
    "Требования к качеству вопросов:\n"
    "1. Вопросы должны проверять ПОНИМАНИЕ материала, а не только запоминание фактов.\n"
    "2. Каждый вопрос должен быть САМОДОСТАТОЧНЫМ — понятным без доступа к исходному "
    "тексту (не используй формулировки вроде «объясните последствия» без указания, "
    "о чём именно идёт речь).\n"
    "3. Дистракторы (неверные варианты) должны быть ПРАВДОПОДОБНЫМИ и различными — "
    "из той же предметной области, без повторений, без заведомо абсурдных вариантов.\n"
    "4. Для single_choice должен существовать ровно один однозначно правильный ответ."
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
        questions = [_dedup_choices(QuizQuestion(**raw.model_dump())) for raw in quiz.questions]
        well_formed = [question for question in questions if question.is_well_formed()]
        if not well_formed:
            return self._refuse(message, reason="llm returned no well-formed quiz questions")
        grounded = [q for q in well_formed if _references_summary(q, summary_text)]
        if not grounded:
            return self._refuse(message, reason="llm returned no grounded quiz questions")
        return self._inform(
            message,
            content={
                "quiz_id": f"quiz-{message.task_id}",
                "questions": [question.model_dump() for question in grounded],
                "difficulty": difficulty,
            },
        )


def _references_summary(question: QuizQuestion, summary_text: str) -> bool:
    """Return True if *question* is grounded in *summary_text*.

    Only ``open_answer`` questions are gated: choice questions already carry
    domain context via their choices, so they always pass.  For open_answer,
    at least one content token (alphanumeric, length >= _MIN_CONTENT_LEN) from
    the question stem must appear in the summary token set.
    """
    if question.type != "open_answer":
        return True
    summary_tokens = {
        token for token in re.findall(r"[a-zа-яё0-9]+", summary_text.lower()) if len(token) >= _MIN_CONTENT_LEN
    }
    question_tokens = [
        token for token in re.findall(r"[a-zа-яё0-9]+", question.question.lower()) if len(token) >= _MIN_CONTENT_LEN
    ]
    return any(token in summary_tokens for token in question_tokens)


def _dedup_choices(question: QuizQuestion) -> QuizQuestion:
    """Return a copy of *question* with duplicate choices removed (case-insensitive,
    whitespace-trimmed), preserving first-occurrence order. answer_idx / answer_indices
    are remapped to the deduped positions so correctness is preserved.

    open_answer questions and questions without choices are returned unchanged.
    """
    if question.type == "open_answer" or not question.choices:
        return question
    new_choices: list[str] = []
    remap: dict[int, int] = {}
    seen: dict[str, int] = {}
    for old_index, choice in enumerate(question.choices):
        key = choice.strip().lower()
        if key in seen:
            remap[old_index] = seen[key]
        else:
            seen[key] = len(new_choices)
            remap[old_index] = seen[key]
            new_choices.append(choice)
    new_answer_idx = remap.get(question.answer_idx) if question.answer_idx is not None else None
    # A multi_choice may lose answer cardinality when two correct options share identical
    # text: after dedup they map to the same index, shrinking answer_indices. Intended —
    # identical options cannot both be shown to a student.
    new_answer_indices = (
        sorted({remap[i] for i in question.answer_indices if i in remap})
        if question.answer_indices is not None
        else None
    )
    return question.model_copy(
        update={"choices": new_choices, "answer_idx": new_answer_idx, "answer_indices": new_answer_indices}
    )


def _summary_to_text(summary: object) -> str:
    if not isinstance(summary, dict):
        return ""
    sections = summary.get("sections")
    if not isinstance(sections, list):
        return ""
    parts = [str(section.get("text", "")) for section in sections if isinstance(section, dict)]
    return "\n".join(part for part in parts if part).strip()
