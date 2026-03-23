"""AML Investigation Agent - Real-time Anti-Money Laundering analysis."""

__all__ = ["create_aml_agent"]


def create_aml_agent(**kwargs):
    """Lazy import to avoid circular imports during data setup."""
    from aml_agent.agent import create_aml_agent as _create
    return _create(**kwargs)
