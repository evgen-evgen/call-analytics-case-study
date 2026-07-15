"""Install or update the MTBank status Pipe through the Open WebUI API."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


FUNCTION_ID = "mtbank_progress"
BASE_URL = os.environ["OPENWEBUI_FUNCTIONS_URL"].rstrip("/")
API_KEY = os.environ["OPENWEBUI_API_KEY"]
CONTENT = Path(__file__).with_name("mtbank_progress.py").read_text()
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    response = urlopen(
        Request(
            f"{BASE_URL}{path}",
            data=data,
            headers=HEADERS,
            method=method,
        ),
        timeout=30,
    )
    return json.loads(response.read() or b"{}")


form = {
    "id": FUNCTION_ID,
    "name": "MTBank Call Analytics",
    "content": CONTENT,
    "meta": {
        "description": (
            "MTBank analysis proxy with replaceable Open WebUI progress status."
        )
    },
}

try:
    current = request("GET", f"/api/v1/functions/id/{FUNCTION_ID}")
except HTTPError as exc:
    if exc.code != 401:
        raise
    current = request("POST", "/api/v1/functions/create", form)
else:
    current = request(
        "POST",
        f"/api/v1/functions/id/{FUNCTION_ID}/update",
        form,
    )

if not current.get("is_active", False):
    current = request(
        "POST",
        f"/api/v1/functions/id/{FUNCTION_ID}/toggle",
    )

model_form = {
    "id": FUNCTION_ID,
    "name": "MTBank Call Analytics",
    "params": {},
    "meta": {
        "description": "Анализ банковских звонков с отображением прогресса."
    },
    "access_grants": [
        {
            "principal_type": "user",
            "principal_id": "*",
            "permission": "read",
        }
    ],
    "is_active": True,
}
try:
    model = request("POST", "/api/v1/models/create", model_form)
except HTTPError as exc:
    if exc.code != 401:
        raise
    model = request("POST", "/api/v1/models/model/update", model_form)

models = request("GET", "/api/v1/configs/models")
default_models = [
    model.strip()
    for model in (models.get("DEFAULT_MODELS") or "").split(",")
    if model.strip() and model.strip() != "mtbank-asr"
]
default_models.insert(0, FUNCTION_ID)
request(
    "POST",
    "/api/v1/configs/models",
    {
        **models,
        "DEFAULT_MODELS": ",".join(dict.fromkeys(default_models)),
        "DEFAULT_PINNED_MODELS": FUNCTION_ID,
    },
)

tasks = request("GET", "/api/v1/tasks/config")
request(
    "POST",
    "/api/v1/tasks/config/update",
    {
        **tasks,
        "ENABLE_TITLE_GENERATION": False,
        "ENABLE_SEARCH_QUERY_GENERATION": False,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": False,
    },
)

audio = request("GET", "/api/v1/audio/config")
request(
    "POST",
    "/api/v1/audio/config/update",
    {
        **audio,
        "stt": {
            **audio["stt"],
            "SUPPORTED_CONTENT_TYPES": [],
        },
    },
)

print(
    json.dumps(
        {
            "event": "openwebui.function.ready",
            "function_id": FUNCTION_ID,
            "active": current.get("is_active", False),
            "public_model": model.get("id") == FUNCTION_ID,
            "default_model": FUNCTION_ID,
        }
    )
)
