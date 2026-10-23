from pathlib import Path


def test_docker_compose_exists():
    assert Path("docker-compose.yml").exists()

def test_project_root_has_required_dirs():
    required = ["agent_host", "mcp_servers", "infra", "ingestion"]
    for path in required:
        assert Path(path).exists(), f"Missing required directory: {path}"