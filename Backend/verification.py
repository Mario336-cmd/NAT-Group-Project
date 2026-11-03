# Connect retrieval, the saved BERT classifier and grounded explanations.

from threading import Lock
from time import perf_counter
import re

from .llm import Explainer
from .languages import LANGUAGE_NAMES, message
from .bert_model import ClaimInputError
from .settings import Settings


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
        language = "en"
        english_claim = None
        evidence = []
        timings = {"language_ms": 0.0, "retrieval_ms": 0.0, "classification_ms": 0.0, "explanation_ms": 0.0}

        def result(status, prediction=None, screening=None, explanation=None, notice=""):
            value = {
                "claim": claim, "language": language, "top_k": limit, "status": status,
                "prediction": prediction, "evidence": evidence,
                "explanation": explanation or {"status": "unavailable", "summary": "", "citations": [], "reason": notice},
                "notice": notice,
                "timings": {**{key: round(value, 2) for key, value in timings.items()},
                            "total_ms": round((perf_counter() - start) * 1000, 2)},
            }
            if english_claim is not None:
                value["english_claim"] = english_claim
            if screening is not None:
                value["screening"] = screening
            return value

        def blocked(status, reason):
            return result("screening_unavailable" if status == "unavailable" else "rejected",
                          screening={"status": status, "reason": reason},
                          explanation={"status": "unavailable", "summary": "", "citations": [], "reason": reason})

        try:
            compact = "".join(claim.split())
            # Reject only unambiguous noise locally; do not guess its language.
            if not any(char.isalnum() for char in claim) or re.fullmatch(r"(.)\1{7,}", compact, flags=re.DOTALL):
                return blocked("invalid_input", message(language, "invalid_input"))
            tick = perf_counter()
            prepared = self.explainer.prepare(claim, configuration=configuration)
            timings["language_ms"] = (perf_counter() - tick) * 1000
            if not isinstance(prepared, dict):
                return blocked("unavailable", message(language, "service_error"))
            detected = prepared.get("language", "en")
            if detected not in LANGUAGE_NAMES:
                return blocked("unavailable", message(language, "service_error"))
            language = detected
            screening = prepared.get("screening")
            if prepared.get("status") != "ready" or not isinstance(screening, dict):
                return blocked("unavailable", prepared.get("reason") or message(language, "service_error"))
            if screening.get("status") in ("invalid_input", "out_of_scope"):
                return blocked(screening["status"], screening.get("reason") or message(language, screening["status"]))
            if screening.get("status") != "accepted":
                return blocked("unavailable", message(language, "service_error"))
            translated = prepared.get("english_claim")
            if not isinstance(translated, str) or not translated.strip() or (language == "en" and translated != claim):
                return blocked("unavailable", message(language, "invalid_output"))
            if len(translated) > self.settings.max_claim_characters:
                return blocked("invalid_input", message(language, "encoding"))
            english_claim = translated
            self.warmup()
            tick = perf_counter()
            # Only the prepared English assertion reaches the existing local models.
            hits = self.retriever.search(english_claim, k=limit)
            evidence = [{**hit, "id": f"E{index}"} for index, hit in enumerate(hits, 1)]
            timings["retrieval_ms"] = (perf_counter() - tick) * 1000
            if not evidence:
                return result("no_evidence", explanation={"status": "unavailable", "summary": "", "citations": [],
                              "reason": message(language, "no_evidence_reason")}, notice=message(language, "no_evidence"))
            tick = perf_counter()
            rendered = "\n".join(dict.fromkeys(hit["rendered"] for hit in evidence))
            try:
                prediction = self.classifier.predict(english_claim, rendered)
            except ClaimInputError:
                return blocked("invalid_input", message(language, "encoding"))
            timings["classification_ms"] = (perf_counter() - tick) * 1000
            tick = perf_counter()
            explanation = self.explainer.explain(claim, evidence, prediction, configuration=configuration,
                                                language=language, english_claim=english_claim)
            timings["explanation_ms"] = (perf_counter() - tick) * 1000
            if not isinstance(explanation, dict):
                return blocked("unavailable", message(language, "service_error"))
            screening = explanation.pop("screening", None)
            if (not isinstance(screening, dict) or screening.get("status") not in ("accepted", "invalid_input", "out_of_scope", "unavailable")
                    or (screening.get("status") == "accepted" and explanation.get("status") != "ready")):
                return blocked("unavailable", message(language, "service_error"))
            if screening["status"] != "accepted":
                reason = (message(language, screening["status"]) if screening["status"] != "unavailable" else
                          explanation.get("reason") or screening.get("reason") or message(language, "service_error"))
                return blocked(screening["status"], reason)
            translations = explanation.pop("evidence_translations", {})
            if language != "en":
                if (not isinstance(translations, dict) or set(translations) != {hit["id"] for hit in evidence}
                        or any(not isinstance(value, str) or not value.strip() or len(value) > 200000 for value in translations.values())):
                    return blocked("unavailable", message(language, "invalid_output"))
                evidence = [{**hit, "translation": translations[hit["id"]]} for hit in evidence]
            explanation.pop("error_code", None)
            return result("prediction", prediction=prediction, screening={"status": "accepted", "reason": None},
                          explanation=explanation, notice=message(language, "notice"))
        except PipelineUnavailableError:
            return blocked("unavailable", message(language, "service_error"))
        except Exception:
            # Never expose local/provider internals or a partial prediction.
            return blocked("unavailable", message(language, "service_error"))
        finally:
            self._lock.release()
