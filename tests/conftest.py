"""Shared pytest fixtures."""
import pytest

from src.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    # scope="session": start Spark once for the whole test run (startup takes seconds).
    session = get_spark("churn-insights-tests")
    yield session
    session.stop()
