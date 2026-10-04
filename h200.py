"""Cliente mínimo de la H200 de la USFQ: solo biblioteca estándar.

Dos servidores, no uno: el LLM en el vLLM (puerto 12555, réplica en 12559) y los
embeddings en el Ollama (11434, `bge-m3`). `/v1/embeddings` contra el 12555 responde 404.

- **No se escribe ningún id de modelo.** Se le pregunta al endpoint (`/v1/models`): el
  laboratorio cambió el modelo servido al menos una vez este semestre.
- **El razonamiento no se apaga.** `enable_thinking: True` solo le pide a vLLM que lo
  SEPARE en `reasoning`; el modelo razona igual y lo cobra en tokens de salida.
- Requiere GlobalProtect. Sin VPN, el constructor falla con un mensaje que lo dice.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class H200:
    HOST = os.environ.get("H200_HOST", "172.28.230.10")
    PUERTO = int(os.environ.get("H200_PUERTO", "12555"))
    PUERTO_EMB = int(os.environ.get("H200_PUERTO_EMB", "11434"))
    MODELO_EMB = os.environ.get("H200_MODELO_EMB", "bge-m3:latest")

    def __init__(self, timeout: float = 600):
        self.timeout = timeout
        self.modelo = self._pedir(self.PUERTO, "v1/models", timeout=8)["data"][0]["id"]

    def _pedir(self, puerto: int, ruta: str, cuerpo: dict | None = None,
               timeout: float | None = None) -> dict:
        pet = urllib.request.Request(
            f"http://{self.HOST}:{puerto}/{ruta}",
            data=None if cuerpo is None else json.dumps(cuerpo).encode(),
            # vLLM no valida la clave; «local» no es una credencial
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(pet, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"La H200 respondió {err.code}: {err.read()[:300]!r}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"Sin respuesta de {self.HOST}:{puerto} ({err.reason}). "
                               "¿Está GlobalProtect conectada?") from err

    def chat(self, mensajes: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, max_tokens: int = 16384,
             temperature: float = 0.0) -> dict:
        """Una llamada. Devuelve el mensaje, el razonamiento separado, el uso y la latencia."""
        cuerpo = {"model": self.modelo, "messages": mensajes, "temperature": temperature,
                  "max_tokens": max_tokens,
                  "chat_template_kwargs": {"enable_thinking": True}}
        if tools:
            cuerpo["tools"] = tools
        if json_mode:
            cuerpo["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = self._pedir(self.PUERTO, "v1/chat/completions", cuerpo)
        eleccion = r["choices"][0]
        m = eleccion["message"]
        return {
            "contenido": m.get("content") or "",
            "tool_calls": m.get("tool_calls") or [],
            "razonamiento": m.get("reasoning") or m.get("reasoning_content") or "",
            "fin": eleccion.get("finish_reason"),
            "uso": r.get("usage", {}),
            "latencia_s": round(time.perf_counter() - t0, 2),
        }

    def embed(self, textos: list[str], lote: int = 64) -> list[list[float]]:
        """Embeddings normalizados (el producto punto es el coseno), por lotes."""
        salida: list[list[float]] = []
        for i in range(0, len(textos), lote):
            r = self._pedir(self.PUERTO_EMB, "v1/embeddings",
                            {"model": self.MODELO_EMB, "input": textos[i:i + lote]})
            for d in sorted(r["data"], key=lambda d: d["index"]):
                v = d["embedding"]
                n = sum(x * x for x in v) ** 0.5 or 1.0
                salida.append([x / n for x in v])
        return salida
