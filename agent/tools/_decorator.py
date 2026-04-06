"""@tool decorator — uses the real Strands decorator when available."""

try:
    from strands import tool
except ImportError:
    # Fallback shim for environments without strands installed
    def tool(fn):
        """Mark *fn* as a Strands-compatible tool (shim)."""
        fn._is_tool = True
        return fn
