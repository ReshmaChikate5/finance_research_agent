import uvicorn
from fastapi import FastAPI

app = FastAPI(title="agent-host (stub)")


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-host"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
