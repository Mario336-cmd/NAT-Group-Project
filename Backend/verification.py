# Connect retrieval, the saved BERT classifier and grounded explanations.

from threading import Lock
from time import perf_counter
import re

from .llm import Explainer, SCREENING_MESSAGES
from .settings import Settings


NOTICE = (
    "BERT prediction based on retrieved Wikipedia passages. "
    "Model confidence is not a probability that a claim is true. "
    "Evidence sufficiency has not yet been calibrated for this application."
)

SETUP_STEPS = {
    "retriever": ("assets", "dependencies", "corpus", "embeddings", "bm25", "validation", "warmup"),
    "classifier": ("dependencies", "tokenizer", "weights", "device", "warmup"),
}


class PipelineBusyError(RuntimeError):
    pass


class PipelineUnavailableError(RuntimeError):
    pass


class VerificationService:
    def __init__(self, settings: Settings, retriever=None, classifier=None, explainer=None):
        self.settings = settings
        self.retriever = retriever
        self.classifier = classifier
        self.explainer = explainer if explainer is not None else Explainer(settings)
        self.state = "not_loaded"
        self._lock = Lock()
        self._load_lock = Lock()
        self._warmed = False
        self._warmup_evidence = None
        self._status_lock = Lock()
        self._components = {
            "retriever": "preparing" if retriever is not None else "waiting",
            "classifier": "preparing" if classifier is not None else "waiting",
        }
        self._completed_steps = {
            name: set(steps[:-1]) if getattr(self, name) is not None else set()
            for name, steps in SETUP_STEPS.items()
        }

    def initialization_status(self):
        with self._status_lock:
            progress = {}
            for name, steps in SETUP_STEPS.items():
                completed = len(self._completed_steps[name])
                progress[name] = {"completed": completed, "total": len(steps), "percent": completed * 100 // len(steps)}
            return {"status": self.state, "components": self._components.copy(), "progress": progress}

    def _complete_setup_step(self, component, step):
        if step not in SETUP_STEPS[component][:-1]:
            raise ValueError("Unknown setup step")
        with self._status_lock:
            self._completed_steps[component].add(step)

    def _progress(self, component=None, status=None, state=None):
        with self._status_lock:
            if component is not None:
                self._components[component] = status
                if status == "ready":
                    self._completed_steps[component].add("warmup")
                elif status == "error":
                    self._completed_steps[component].discard("warmup")
            if state is not None:
                self.state = state
                if state == "error":
                    for name, current in self._components.items():
                        if current in ("loading", "preparing"):
                            self._components[name] = "waiting"

    def queue_warmup(self):
        with self._status_lock:
            if self.state != "ready":
                self.state = "loading"
                for component, status in self._components.items():
                    if status == "error":
                        self._components[component] = "waiting"
                        if getattr(self, component) is None:
                            self._completed_steps[component].clear()

    def _initialize(self):
        self._progress(state="loading")
        component = "retriever"
        try:
            if self.retriever is None:
                self._progress(component, "loading")
                from .retrieval import EvidenceRetriever
                self.retriever = EvidenceRetriever(
                    self.settings, progress=lambda step: self._complete_setup_step("retriever", step)
                )
            self._progress(component, "ready" if "warmup" in self._completed_steps[component] else "preparing")
            component = "classifier"
            if self.classifier is None:
                self._progress(component, "loading")
                from .bert_model import BertClassifier
                self.classifier = BertClassifier(
                    self.settings, progress=lambda step: self._complete_setup_step("classifier", step)
                )
            self._progress(component, "ready" if "warmup" in self._completed_steps[component] else "preparing")
        except Exception as exc:
            self._progress(component, "error", state="error")
            raise PipelineUnavailableError(
                "The local retrieval or BERT assets could not be loaded. Check the server setup and retry."
            ) from exc

    def warmup(self):
        with self._load_lock:
            if self._warmed and self.state == "ready":
                return
            self._initialize()
            # Initialize local kernels without calling the explanation provider.
            component = "retriever"
            try:
                claim = "Windhoek is the capital of Namibia."
                if self._warmup_evidence is None or "warmup" not in self._completed_steps[component]:
                    self._progress(component, "preparing")
                    hits = self.retriever.search(claim, k=1)
                    self._warmup_evidence = "\n".join(dict.fromkeys(hit["rendered"] for hit in hits)) or claim
                    self._progress(component, "ready")
                component = "classifier"
                self._progress(component, "preparing")
                self.classifier.predict(claim, self._warmup_evidence)
                self._warmed = True
                self._progress(component, "ready", state="ready")
            except Exception:
                self._progress(component, "error", state="error")
                raise

    def verify(self, claim: str, top_k: int | None = None, configuration=None) -> dict:
        claim = claim.strip()
        if not claim or len(claim) > self.settings.max_claim_characters:
            raise ValueError(f"Enter a claim of 1–{self.settings.max_claim_characters} characters.")
        limit = self.settings.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10:
            raise ValueError("Choose a passage count from 1 to 10.")
        if not self._lock.acquire(blocking=False):
            raise PipelineBusyError("Another claim is still being processed. Please wait and retry.")
        start = perf_counter()
        try:
            compact = "".join(claim.split())
            # Reject only unambiguous noise locally. Semantic screening uses the LLM.
            if not any(char.isalnum() for char in claim) or re.fullmatch(r"(.)\1{7,}", compact, flags=re.DOTALL):
                reason = SCREENING_MESSAGES["invalid_input"]
                return {
                    "claim": claim, "top_k": limit, "status": "rejected", "prediction": None, "evidence": [],
                    "screening": {"status": "invalid_input", "reason": reason},
                    "explanation": {"status": "unavailable", "summary": "", "citations": [], "reason": reason},
                    "notice": "",
                    "timings": {"retrieval_ms": 0.0, "classification_ms": 0.0, "explanation_ms": 0.0,
                                "total_ms": round((perf_counter() - start) * 1000, 2)},
                }
            self.warmup()
            tick = perf_counter()
            hits = self.retriever.search(claim, k=limit)
            evidence = [{**hit, "id": f"E{index}"} for index, hit in enumerate(hits, 1)]
            retrieval_ms = (perf_counter() - tick) * 1000
            if not evidence:
                return {
                    "claim": claim, "top_k": limit, "status": "no_evidence", "prediction": None, "evidence": [],
                    "explanation": {"status": "unavailable", "summary": "", "citations": [],
                                    "reason": "No passages were retrieved; no model prediction was generated."},
                    "notice": "No evidence was retrieved. This claim could not be evaluated.",
                    "timings": {"retrieval_ms": round(retrieval_ms, 2), "classification_ms": 0.0,
                                "explanation_ms": 0.0, "total_ms": round((perf_counter() - start) * 1000, 2)},
                }
            tick = perf_counter()
            # Match preparation: one rendered passage per newline; IDs are only for citations.
            rendered = "\n".join(dict.fromkeys(hit["rendered"] for hit in evidence))
            prediction = self.classifier.predict(claim, rendered)
            classification_ms = (perf_counter() - tick) * 1000
            tick = perf_counter()
            explanation = self.explainer.explain(claim, evidence, prediction, configuration=configuration)
            explanation_ms = (perf_counter() - tick) * 1000
            screening = explanation.pop("screening", None)
            # Missing screening metadata cannot silently authorize a new prediction.
            if (not isinstance(screening, dict) or screening.get("status") not in ("accepted", "invalid_input", "out_of_scope", "unavailable")
                    or (screening.get("status") == "accepted" and explanation.get("status") != "ready")):
                screening = {"status": "unavailable", "reason": "Input screening could not be completed. Please retry."}
            if screening["status"] != "accepted":
                reason = SCREENING_MESSAGES.get(screening["status"], screening.get("reason") or "Input screening could not be completed. Please retry.")
                screening = {"status": screening["status"], "reason": reason}
                explanation = {"status": "unavailable", "summary": "", "citations": [], "reason": reason}
                prediction = None
            return {
                "claim": claim, "top_k": limit,
                "status": "prediction" if prediction is not None else "screening_unavailable" if screening["status"] == "unavailable" else "rejected",
                "prediction": prediction, "screening": screening,
                "evidence": evidence, "explanation": explanation, "notice": NOTICE if prediction is not None else "",
                "timings": {"retrieval_ms": round(retrieval_ms, 2),
                            "classification_ms": round(classification_ms, 2),
                            "explanation_ms": round(explanation_ms, 2),
                            "total_ms": round((perf_counter() - start) * 1000, 2)},
            }
        finally:
            self._lock.release()
