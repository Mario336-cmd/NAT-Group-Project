# Transcribe bounded browser recordings with the saved OpenAI configuration.

from contextlib import nullcontext

from .settings import Settings


class SpeechError(ValueError):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class SpeechTranscriber:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client

    def transcribe(self, data, media_type, configuration=None):
        if not data:
            raise SpeechError(422, "The recording is empty. Record your claim again.")
        if len(data) > self.settings.max_audio_bytes:
            raise SpeechError(413, "Keep recordings to 30 seconds and 5 MiB or fewer.")
        media_type = media_type.split(";", 1)[0].strip().lower()
        if media_type in ("audio/webm", "video/webm") and data.startswith(b"\x1a\x45\xdf\xa3"):
            filename, content_type = "recording.webm", "audio/webm"
        elif media_type in ("audio/mp4", "audio/m4a", "audio/x-m4a", "video/mp4") and data[4:8] == b"ftyp":
            filename, content_type = "recording.m4a", "audio/mp4"
        elif media_type in ("audio/wav", "audio/x-wav") and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            filename, content_type = "recording.wav", "audio/wav"
        else:
            raise SpeechError(415, "The recording format could not be read. Use a browser that records WebM, M4A or WAV audio.")
        if configuration is None:
            raise SpeechError(503, "Configure Voice input in Settings before recording.")
        if configuration.provider != "openai" or configuration.purpose != "transcription":
            raise SpeechError(503, "Voice input requires an OpenAI transcription configuration.")
        try:
            import openai
        except ImportError:
            raise SpeechError(503, "Install the backend requirements to enable voice input.") from None
        try:
            context = nullcontext(self._client) if self._client is not None else configuration.clients.use(
                configuration, self.settings.transcription_timeout
            )
            with context as client:
                result = client.audio.transcriptions.create(
                    model=configuration.model, file=(filename, data, content_type),
                    timeout=self.settings.transcription_timeout,
                )
            text = getattr(result, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise SpeechError(422, "No speech was detected. Record your claim again.")
            text = text.strip()
            if len(text) > self.settings.max_claim_characters:
                raise SpeechError(422, "The transcript exceeds 4,000 characters. Record a shorter claim.")
            return {"text": text, "model": configuration.model}
        except SpeechError:
            raise
        except openai.AuthenticationError:
            detail = "OpenAI rejected the voice input API key. Update it in Settings."
        except (openai.PermissionDeniedError, openai.NotFoundError):
            detail = "The transcription model is unavailable for this key. Check the model name and access in Settings."
        except openai.RateLimitError:
            detail = "OpenAI's rate limit or API quota was reached. Check usage and billing, then retry."
        except openai.APITimeoutError:
            detail = "Transcription timed out. Your draft was kept. Try recording again."
        except openai.APIConnectionError:
            detail = "Could not connect to OpenAI for transcription. Check the internet connection."
        except openai.BadRequestError:
            detail = "OpenAI could not transcribe this recording. Check that the model supports audio transcription."
        except Exception:
            detail = "Voice transcription failed. Your draft was kept. Try again or type the claim."
        raise SpeechError(503, detail) from None
