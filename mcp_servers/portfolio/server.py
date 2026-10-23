from fastapi import FastAPI
import uvicorn

app = FastAPI(title="mcp-portfolio (stub)")


@app.get("/health")
def health():
    return {"status": "ok", "service": "mcp-portfolio"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
