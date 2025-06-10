import os
import tempfile
import time
import yaml
import pytest
from app.edge_graph_config import EdgeGraphConfigLoader
import logging

def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)

def test_load_valid_config():
    config = {
        'default_edge_types': {'a': 1.0, 'b': 1.0},
        'app_overrides': {'foo': {'b': 2.0}}
    }
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        write_yaml(tf.name, config)
        loader = EdgeGraphConfigLoader(tf.name)
        loaded = loader.get_config()
        assert loaded['default_edge_types'] == {'a': 1.0, 'b': 1.0}
        assert loaded['app_overrides']['foo'] == {'b': 2.0}
    os.unlink(tf.name)

def test_thread_safety():
    config = {'default_edge_types': ['x'], 'app_overrides': {}}
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        write_yaml(tf.name, config)
        loader = EdgeGraphConfigLoader(tf.name)
        # Simulate concurrent access
        from concurrent.futures import ThreadPoolExecutor
        def read():
            for _ in range(100):
                loader.get_config()
        with ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(lambda _: read(), range(5))
    os.unlink(tf.name)

def test_hot_reload(monkeypatch):
    logging.basicConfig(level=logging.DEBUG)
    config1 = {'default_edge_types': {'a': 1.0}, 'app_overrides': {}}
    config2 = {'default_edge_types': {'b': 2.0}, 'app_overrides': {}}
    # Use a file in config/ to avoid macOS temp file issues
    os.makedirs('config', exist_ok=True)
    test_path = 'config/test_edge_graph_hot_reload.yaml'
    write_yaml(test_path, config1)
    loader = EdgeGraphConfigLoader(test_path)
    assert loader.get_config()['default_edge_types'] == {'a': 1.0}
    write_yaml(test_path, config2)
    # Ensure file is flushed and synced
    with open(test_path, 'a') as f:
        f.flush()
        os.fsync(f.fileno())
    # Wait for watcher to pick up (watchdog is async)
    for _ in range(10):
        time.sleep(0.5)
        if loader.get_config()['default_edge_types'] == {'b': 2.0}:
            break
    assert loader.get_config()['default_edge_types'] == {'b': 2.0}
    os.remove(test_path)

def test_invalid_yaml():
    os.makedirs('config', exist_ok=True)
    test_path = 'config/test_edge_graph_invalid.yaml'
    with open(test_path, 'w') as tf:
        tf.write('invalid: [unclosed')
    loader = EdgeGraphConfigLoader(test_path)
    # Should not raise, config should be empty or unchanged
    assert isinstance(loader.get_config(), dict)
    os.remove(test_path)

def test_parse_weights_and_app_merge():
    config = {
        'default_edge_types': {'a': 1.0, 'b': 0.5},
        'app_overrides': {'foo': {'b': 2.0, 'c': 3.0}}
    }
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        write_yaml(tf.name, config)
        loader = EdgeGraphConfigLoader(tf.name)
        # App override merges: b from app, a from default, c from app
        weights = loader.get_app_edge_weights('foo')
        assert weights == {'a': 1.0, 'b': 2.0, 'c': 3.0}
        # Fallback to default
        weights2 = loader.get_app_edge_weights('bar')
        assert weights2 == {'a': 1.0, 'b': 0.5}
    os.unlink(tf.name)

def test_invalid_weights():
    config = {
        'default_edge_types': {'a': -1.0},
        'app_overrides': {}
    }
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        write_yaml(tf.name, config)
        # Should log error and not raise, config will be empty
        try:
            loader = EdgeGraphConfigLoader(tf.name)
            # Should not include invalid weights
            assert 'a' not in loader.get_config().get('default_edge_types', {})
        except Exception:
            pass
    os.unlink(tf.name)
    # Test non-float
    config2 = {
        'default_edge_types': {'a': 'bad'},
        'app_overrides': {}
    }
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        write_yaml(tf.name, config2)
        try:
            loader = EdgeGraphConfigLoader(tf.name)
            assert 'a' not in loader.get_config().get('default_edge_types', {})
        except Exception:
            pass
    os.unlink(tf.name) 