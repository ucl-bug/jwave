During developement / debugging, one way to inspect the intermediate state of variables inside compiled functions is to disable `jit` compilation globally

```python
from jax.config import config
config.update('jax_disable_jit', True)
```

### Run tests
Before merging with a main branch, such as the master, run the tests and generate the badges via
```bash
./test_and_badges_commit.sh
```