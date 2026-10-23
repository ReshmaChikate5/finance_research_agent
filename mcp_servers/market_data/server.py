import uvicorn
from fastapi import FastAPI

app = FastAPI(title="mcp-marketdata (stub)")


@app.get("/health")
def health():
    return {"status": "ok", "service": "mcp-marketdata"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
