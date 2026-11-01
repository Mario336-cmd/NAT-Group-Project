# Session configuration with Windows credential storage and reusable API clients.

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import re
import secrets
from threading import RLock
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .settings import Settings


COOKIE_NAME = "truecheck_session"
PROFILE_COOKIE_NAME = "truecheck_profile"
PROVIDERS = ("openai", "anthropic")
SLOT_PURPOSES = {1: "explanation", 2: "extraction", 3: "transcription"}
PURPOSES = tuple(SLOT_PURPOSES.values())


class PersistenceError(RuntimeError):
    pass


class ProviderClients:
    def __init__(self):
        self._lock = RLock()
        self._clients = {}
        self._users = 0
        self._retired = False

    @contextmanager
    def use(self, configuration, timeout):
        with self._lock:
            if self._retired:
                raise ValueError("This API configuration was cleared or replaced.")
            if timeout not in self._clients:
                if configuration.provider == "openai":
                    from openai import OpenAI as factory
                    base_url = "https://api.openai.com/v1"
                else:
                    from anthropic import Anthropic as factory
                    base_url = "https://api.anthropic.com"
                self._clients[timeout] = factory(
                    api_key=configuration.api_key.get_secret_value(), base_url=base_url,
                    timeout=timeout, max_retries=0,
                )
            client = self._clients[timeout]
            self._users += 1
        try:
            yield client
        finally:
            with self._lock:
                self._users -= 1
                if self._retired and not self._users:
                    self._close_clients()

    def _close_clients(self):
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()

    def close(self):
        with self._lock:
            self._retired = True
            if not self._users:
                self._close_clients()


class ConfigurationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai", "anthropic"]
    purpose: Literal["explanation", "extraction", "transcription"]
    model: str = Field(min_length=1, max_length=128, strict=True)
    api_key: SecretStr | None = Field(default=None, max_length=512)

    @field_validator("model")
    @classmethod
    def valid_model(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) or value.startswith("sk-"):
            raise ValueError("Enter a model name using letters, numbers, dots, colons, underscores or hyphens.")
        return value

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value):
        if value is None:
            return None
        key = value.get_secret_value().lstrip("\ufeff").strip()
        if not re.fullmatch(r"sk-[A-Za-z0-9_-]{17,509}", key):
            raise ValueError("Enter one API key without labels or whitespace.")
        return SecretStr(key)

    @model_validator(mode="after")
    def compatible_configuration(self):
        if self.purpose == "transcription" and self.provider != "openai":
            raise ValueError("Voice input requires an OpenAI transcription model.")
        if self.api_key is not None:
            is_anthropic = self.api_key.get_secret_value().startswith("sk-ant-")
            if is_anthropic != (self.provider == "anthropic"):
                raise ValueError("The key format does not match the selected provider.")
        return self


@dataclass(frozen=True)
class APIConfiguration:
    provider: str
    purpose: str
    model: str
    api_key: SecretStr = field(repr=False)
    clients: ProviderClients = field(default_factory=ProviderClients, repr=False, compare=False)


class WindowsConfigurationStore:
    def __init__(self, settings: Settings, vault=None):
        # Explicitly choose the Windows backend. Never fall back to a text keyring.
        self._vault = vault
        namespace = hashlib.sha256(str(settings.root.resolve()).casefold().encode()).hexdigest()[:16]
        self._service = "TrueCheck-" + namespace

    def _backend(self):
        if self._vault is None:
            from keyring.backends.Windows import WinVaultKeyring
            self._vault = WinVaultKeyring()
        return self._vault

    def _name(self, profile_id, slot):
        if not isinstance(profile_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", profile_id):
            raise PersistenceError("The saved settings profile is invalid. Reopen Settings.")
        return hashlib.sha256(profile_id.encode()).hexdigest() + "-" + str(slot)

    def load(self, profile_id):
        slots = {}
        try:
            for slot, purpose in SLOT_PURPOSES.items():
                raw = self._backend().get_password(self._service, self._name(profile_id, slot))
                if raw is None:
                    continue
                if len(raw) > 4096:
                    raise ValueError("Invalid credential size")
                supplied = ConfigurationInput.model_validate(json.loads(raw))
                if supplied.purpose != purpose or supplied.api_key is None:
                    raise ValueError("Invalid saved configuration")
                slots[slot] = APIConfiguration(supplied.provider, supplied.purpose, supplied.model, supplied.api_key)
            return slots
        except Exception:
            raise PersistenceError("Saved API settings could not be restored from Windows Credential Manager. Check the backend requirements or clear and save the settings again.") from None

    def save(self, profile_id, slot, configuration):
        try:
            payload = {"provider": configuration.provider, "purpose": configuration.purpose,
                       "model": configuration.model, "api_key": configuration.api_key.get_secret_value()}
            self._backend().set_password(self._service, self._name(profile_id, slot), json.dumps(payload))
        except Exception:
            raise PersistenceError("API settings could not be saved in Windows Credential Manager. The previous settings were kept. Check the backend requirements and retry.") from None

    def clear(self, profile_id, slot=None):
        failed = False
        for number in ([slot] if slot is not None else SLOT_PURPOSES):
            try:
                from keyring.errors import PasswordDeleteError
                try:
                    self._backend().delete_password(self._service, self._name(profile_id, number))
                except PasswordDeleteError:
                    # Missing entries are already cleared.
                    if self._backend().get_password(self._service, self._name(profile_id, number)) is not None:
                        raise
            except Exception:
                failed = True
        if failed:
            raise PersistenceError("Some saved API settings could not be removed from Windows Credential Manager. Retry Clear before closing the app.") from None


@dataclass
class Session:
    token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    created: float
    last_used: float
    slots: dict[int, APIConfiguration] = field(default_factory=dict, repr=False)
    profile_id: str | None = field(default=None, repr=False)
    persistence_error: str = ""


class SessionStore:
    def __init__(self, settings: Settings, clock=monotonic, persistence=None):
        self.settings = settings
        self.clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        self.persistence = persistence or WindowsConfigurationStore(settings)

    def _release(self, session):
        for config in session.slots.values():
            config.clients.close()
        session.slots.clear()

    def _expire(self):
        now = self.clock()
        for token, session in list(self._sessions.items()):
            if (now - session.last_used >= self.settings.session_idle_seconds
                    or now - session.created >= self.settings.session_max_seconds):
                self._release(session)
                del self._sessions[token]

    def cleanup(self):
        with self._lock:
            self._expire()

    def create(self, profile_id=None, persistent=False):
        with self._lock:
            self._expire()
            if len(self._sessions) >= self.settings.max_sessions:
                raise RuntimeError("The session limit has been reached. Please retry later.")
            now = self.clock()
            session = Session(secrets.token_urlsafe(32), secrets.token_urlsafe(32), now, now)
            if persistent:
                session.profile_id = profile_id if isinstance(profile_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{43}", profile_id) else secrets.token_urlsafe(32)
                try:
                    session.slots = self.persistence.load(session.profile_id)
                except PersistenceError as exc:
                    session.persistence_error = str(exc)
            self._sessions[session.token] = session
            return session

    def get(self, token):
        with self._lock:
            self._expire()
            return self._sessions.get(token) if isinstance(token, str) else None

    def authorize(self, token, csrf_token):
        with self._lock:
            session = self.get(token)
            if (session is None or not isinstance(csrf_token, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{43}", csrf_token)
                    or not secrets.compare_digest(session.csrf_token, csrf_token)):
                return None
            session.last_used = self.clock()
            return session

    def _require_live(self, session):
        if self.get(session.token) is not session:
            raise ValueError("This settings session expired. Reopen Settings and enter the keys again.")

    def status(self, session):
        with self._lock:
            self._require_live(session)
            slots = []
            for slot in SLOT_PURPOSES:
                config = session.slots.get(slot)
                slots.append({"slot": slot, "configured": config is not None,
                              "provider": config.provider if config else None,
                              "purpose": SLOT_PURPOSES[slot],
                              "model": config.model if config else ""})
            return {"slots": slots, "csrf_token": session.csrf_token,
                    "persistent": session.profile_id is not None,
                    "persistence_error": session.persistence_error,
                    "idle_timeout_seconds": self.settings.session_idle_seconds,
                    "supported": [{"provider": provider, "purpose": purpose}
                                  for provider in PROVIDERS for purpose in PURPOSES
                                  if purpose != "transcription" or provider == "openai"]}

    def save(self, session, slot, supplied):
        with self._lock:
            self._require_live(session)
            if slot not in SLOT_PURPOSES:
                raise ValueError("Choose configuration 1, 2 or 3.")
            if supplied.purpose != SLOT_PURPOSES[slot]:
                raise ValueError("Configuration purposes are fixed: explanations, image/PDF extraction and voice input.")
            existing = session.slots.get(slot)
            key = supplied.api_key
            if key is None:
                if existing is None or existing.provider != supplied.provider:
                    raise ValueError("Enter an API key for the selected provider.")
                key = existing.api_key
            configuration = APIConfiguration(supplied.provider, supplied.purpose, supplied.model, key)
            if session.profile_id is not None:
                self.persistence.save(session.profile_id, slot, configuration)
            # Invalidate older sessions belonging to the same browser profile too.
            for target in self._sessions.values():
                if target is session or (session.profile_id is not None and target.profile_id == session.profile_id):
                    old = target.slots.get(slot)
                    if old is not None:
                        old.clients.close()
                    target.slots[slot] = configuration if target is session else APIConfiguration(
                        supplied.provider, supplied.purpose, supplied.model, key
                    )
                    target.persistence_error = ""

    def for_purpose(self, session, purpose):
        with self._lock:
            if session is None or self.get(session.token) is not session:
                return None
            return next((config for config in session.slots.values() if config.purpose == purpose), None)

    def clear(self, session, slot=None):
        with self._lock:
            self._require_live(session)
            if session.profile_id is not None:
                self.persistence.clear(session.profile_id, slot)
            for target in self._sessions.values():
                if target is session or (session.profile_id is not None and target.profile_id == session.profile_id):
                    if slot is None:
                        self._release(target)
                    else:
                        config = target.slots.pop(slot, None)
                        if config is not None:
                            config.clients.close()
                    target.persistence_error = ""
            if slot is None:
                del self._sessions[session.token]

    def close(self):
        with self._lock:
            for session in self._sessions.values():
                self._release(session)
            self._sessions.clear()
