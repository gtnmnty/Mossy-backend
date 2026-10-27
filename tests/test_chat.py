from fastapi.testclient import TestClient

from app.main import app

def test_rejects_assistant_last():
    r = TestClient(app).post("/chat", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 422
