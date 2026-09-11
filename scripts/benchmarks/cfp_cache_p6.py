"""P6 diagnostics for documentation and an isolated installed wheel.

No bootstrap changes escape this process or its spawned release-smoke worker.
The wheel check never adds the checkout's Python/native package to sys.path.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

# This machine has an unrelated editable checkout; also remove it on spawn.
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
ROOT = Path(__file__).resolve().parents[2]


def inspect_archives(output):
    import hashlib, tarfile, zipfile
    sha = lambda b: hashlib.sha256(b).hexdigest()
    wheel, = (ROOT/'build/cfp-cache-p6-dist').glob('*.whl')
    sdist, = (ROOT/'build/cfp-cache-p6-dist').glob('*.tar.gz')
    with zipfile.ZipFile(wheel) as z, tarfile.open(sdist) as t:
        names, members = z.namelist(), t.getnames()
        prefix = members[0].split('/')[0]+'/'
        required = []
        for relative in ('preparation', 'storage'):
            required.extend((ROOT/f'mtlearn/python/mtlearn/layers/cfp/{relative}').glob('*.py'))
        required += [ROOT/'mtlearn/python/mtlearn/layers/cfp/normalization/statistics_snapshot.py',
                     ROOT/'mtlearn/python/mtlearn/layers/cfp/runtime/_prepared_lifetime.py']
        for path in required:
            pkg = str(path.relative_to(ROOT/'mtlearn/python'))
            source = str(path.relative_to(ROOT))
            assert z.read(pkg) == path.read_bytes(), pkg
            assert t.extractfile(prefix+source).read() == path.read_bytes(), source
        source_modules = [p for p in (ROOT/'mtlearn/python/mtlearn').rglob('*.py')
                          if p.name != '_version.py']  # generated distribution metadata
        for path in source_modules:
            assert z.read(str(path.relative_to(ROOT/'mtlearn/python'))) == path.read_bytes()
            assert t.extractfile(prefix+str(path.relative_to(ROOT))).read() == path.read_bytes()
        for relative in ('scripts/smoke_test_release.py',
                         'mtlearn/tests/python/test_cfp_checkpoint_independence.py'):
            assert t.extractfile(prefix+relative).read() == (ROOT/relative).read_bytes()
        for path in (ROOT/'mtlearn/tests/python/fixtures/cfp_cache_p0').iterdir():
            if path.is_file():
                assert t.extractfile(prefix+str(path.relative_to(ROOT))).read() == path.read_bytes()
        forbidden = ('notebooks/', 'dat/', '__pycache__', '.ipynb_checkpoints',
                     'backend-internalization', 'public-snapshot-checklist',
                     'connected-filter-preprocessing', 'research-roadmap')
        bad = [name for name in members if any(part in name for part in forbidden)]
        assert not bad, bad[:10]
        native, = [name for name in names if name.startswith('mtlearn/_mtlearn.') and name.endswith('.so')]
        manifest = {'wheel': {'path': str(wheel.relative_to(ROOT)), 'sha256': sha(wheel.read_bytes()),
                              'bytes': wheel.stat().st_size, 'members': len(names),
                              'native_sha256': sha(z.read(native))},
                    'sdist': {'path': str(sdist.relative_to(ROOT)), 'sha256': sha(sdist.read_bytes()),
                              'bytes': sdist.stat().st_size, 'members': len(members)},
                    'new_modules_verified': [str(p.relative_to(ROOT/'mtlearn/python')) for p in required],
                    'source_python_modules_verified': len(source_modules),
                    'legacy_fixtures_in_sdist': True, 'release_exclusions_passed': True,
                    'wheel_built_from_sdist': True}
    (output/'package-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['docs', 'wheel', 'examples', 'archives'])
    parser.add_argument('--mock-runtime', action='store_true')
    parser.add_argument('--install-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'examples':
        import re
        import subprocess
        import textwrap
        blocks = re.findall(r'```python\n(.*?)```',
                            (ROOT/'docs/source/guides/cfp-cache-migration.md').read_text(), re.S)
        assert len(blocks) == 3
        bootstrap = f"""import sys
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_mtlearn_editable']
sys.path[:0] = [{str(ROOT/'mtlearn/python')!r}, {str(ROOT/'build/cfp-cache-p0-release/mtlearn/bindings')!r}]
if __name__ == '__main__':
    import torch
    from mtlearn import morphology
    from mtlearn.layers import ConnectedFilterPreprocessingLayer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    images = torch.arange(96, dtype=torch.float32).reshape(3, 1, 4, 8)
    train_dataset = torch.utils.data.TensorDataset(images, images / 255)
    layer = ConnectedFilterPreprocessingLayer(1, [{{'tree_type': 'max-tree', 'attributes': (morphology.AttributeType.AREA,)}}], scale_mode='dataset_zscore')
"""
        script = bootstrap + '\n'.join(textwrap.indent(block, '    ') for block in blocks)
        script += ("\n    assert torch.isfinite(response).all()\n    response.sum().backward()\n"
                   "    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())\n"
                   "    print('All three migration examples and final backward passed')\n")
        path = args.output.resolve()/'guide-examples.py'
        path.write_text(script)
        raise SystemExit(subprocess.call([sys.executable, str(path)], cwd=args.output))
    if args.command == 'archives':
        inspect_archives(args.output)
        return
    if args.command == 'docs':
        overrides = {}
        if args.mock_runtime:
            overrides['autodoc_mock_imports'] = ['torch', 'numpy', 'cv2', '_mtlearn', 'mtlearn._mtlearn']
        else:
            import cfp_cache_p0 as p0
            p0.runtime('build/cfp-cache-p0-release')
        from sphinx.application import Sphinx
        app = Sphinx(str(ROOT/'docs/source'), str(ROOT/'docs/source'),
                     str(args.output/'html'), str(args.output/'doctrees'), 'html',
                     confoverrides=overrides, freshenv=True, warningiserror=True)
        app.build(force_all=True)
        raise SystemExit(app.statuscode)

    assert args.install_root is not None
    install = args.install_root.resolve()
    sys.path.insert(0, str(install))
    import torch
    import mtlearn
    from mtlearn._native import load_bindings
    native = load_bindings()
    assert Path(mtlearn.__file__).resolve().is_relative_to(install), mtlearn.__file__
    assert Path(native.__file__).resolve().is_relative_to(install), native.__file__
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('release_smoke', ROOT/'scripts/smoke_test_release.py')
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    saved = sys.argv
    sys.argv = [str(spec.origin), '--expected-version', mtlearn.__version__]
    try:
        assert smoke.main() == 0
    finally:
        sys.argv = saved
    import pytest
    status = pytest.main(['-q', str(ROOT/'mtlearn/tests/python/test_cfp_checkpoint_independence.py'),
                          '--junitxml='+str(args.output/'checkpoint-tests.xml')])
    import hashlib
    result = {
        'package_path': mtlearn.__file__, 'native_path': native.__file__,
        'native_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        'version': mtlearn.__version__, 'python': sys.version, 'torch': str(torch.__version__),
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'release_smoke': 'passed', 'checkpoint_pytest_exit': int(status),
        'source_package_used': False,
    }
    (args.output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
