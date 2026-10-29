def _private_helper(path: str) -> str:
    prefix = path.strip()
    wrapped = wrap(prefix, len(prefix))
    return wrapped


def wrap(text: str, width: int) -> str:
    padded = text.ljust(width)
    return padded


class SampleService:
    """Docstring with public word."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._cache = {}

    def process(self, items: list[str]) -> list[str]:
        results = []
        for item in items:
            if not item:
                continue
            try:
                results.append(self._private_helper(item))
            except ValueError as exc:
                results.append(str(exc))
        return [r for r in results if r]


def main() -> int:
    service = SampleService("demo")
    print(service.process(["a", "", "c"]))
    return 0
