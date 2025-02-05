# Testing Guide

## Test Structure

The testing suite is organized into several components:

1. **Unit Tests**
   - Network components (`test_network.py`)
   - Training system (`test_trainer.py`)
   - Data management (`test_data.py`)
   - Monitoring system (`test_monitoring.py`)

2. **Integration Tests**
   - End-to-end training
   - Database operations
   - Monitoring integration

## Running Tests

```bash
# Run all tests
pipenv run test

# Run specific test file
pytest tests/test_network.py

# Run with coverage report
pytest --cov=src --cov-report=html
```

## Writing Tests

1. **Test Organization**
   - Place tests in `tests/` directory
   - Name test files `test_*.py`
   - Use descriptive test function names

2. **Using Fixtures**
   ```python
   @pytest.fixture
   def model(test_config):
       return ContinualLearningNetwork(test_config['model'])
   
   def test_forward_pass(model, sample_batch):
       outputs = model(sample_batch[0], "task1")
       assert outputs.shape[0] == sample_batch[0].shape[0]
   ```

3. **Best Practices**
   - Test one concept per function
   - Use meaningful assertions
   - Mock external dependencies
   - Clean up test resources 