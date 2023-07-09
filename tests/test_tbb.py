import pytest
from pydin.core.tbb import TBBControl


def test_tbb_control_max_allowed_parallelism():
    tbb_control = TBBControl()

    # Initial value is hardware concurrency
    assert tbb_control.max_allowed_parallelism == tbb_control.hardware_concurrency()

    # Test setting a new value
    tbb_control.max_allowed_parallelism = 4
    assert tbb_control.max_allowed_parallelism == 4

    # Test setting it back
    tbb_control.max_allowed_parallelism = tbb_control.hardware_concurrency()
    assert tbb_control.max_allowed_parallelism == tbb_control.hardware_concurrency()


def test_tbb_control_thread_stack_size():
    tbb_control = TBBControl()
    stack_size = tbb_control.get_thread_stack_size()
    assert isinstance(stack_size, int)  # The actual value may vary depending on the system settings


def test_tbb_context_manager():
    with TBBControl() as tbb_control:
        assert tbb_control.max_allowed_parallelism == tbb_control.hardware_concurrency()

        # Test setting a new value inside context manager
        tbb_control.max_allowed_parallelism = 4
        assert tbb_control.max_allowed_parallelism == 4


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
