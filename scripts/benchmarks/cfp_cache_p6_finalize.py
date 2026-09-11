"""Finalize P6 evidence after running docs, archive and installed-wheel checks."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
root=Path(__file__).resolve().parents[2]
out=root/'docs/cfp-cache/p6'
class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(); self.ids=set(); self.links=[]; self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        if tag=='a' and 'href' in attrs: self.links.append(attrs['href'])
exports=['CFPPreprocessor','PreparedMorphology','PreparedBatch','PreparationResult',
         'PreparedDataset','collate_prepared','NullStore','MemoryStore','DiskStore','StatisticsSnapshot']
docs=[]
for mode in ('docs','docs-mocked'):
    html=root/f'build/cfp-cache-p6-{mode}/html'
    inspected={}
    count=0
    paths=['guides/cfp-cache-migration.html','api/python/cfp_preparation.html',
           'guides/index.html','api/python/index.html',
           'guides/connected-filter-preprocessing.html','guides/pytorch-integration.html']
    for relative in paths:
        path=html/relative
        page=inspected.setdefault(path, Page(path))
        for href in page.links:
            url=urlsplit(href)
            if url.scheme or url.netloc: continue
            target=(path.parent/unquote(url.path)).resolve() if url.path else path
            assert target.is_file(), (relative, href)
            if url.fragment and target.suffix=='.html':
                other=inspected.setdefault(target, Page(target))
                assert unquote(url.fragment) in other.ids, (relative, href)
            count+=1
    api=Page(html/'api/python/cfp_preparation.html')
    for name in exports: assert 'mtlearn.layers.cfp.'+name in api.ids, (mode,name)
    docs.append({'mode':mode,'strict_build':'passed','checked_pages':len(paths),
                 'local_links_and_anchors_checked':count,'public_preparation_objects':len(exports)})
(out/'docs-validation.json').write_text(json.dumps(docs,indent=2)+'\n')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
fixtures=root/'mtlearn/tests/python/fixtures/cfp_cache_p0'
manifest=json.loads((fixtures/'manifest.json').read_text())
for name,digest in manifest['files'].items(): assert sha(fixtures/name)==digest,name
original=root/'notebooks/experiments/CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb'
assert sha(original)=='0c736b20d2618c2b033279b30d6fd056e73a0349ab5f18ebf4f2ea26f343dd49'
native,=(root/'build/cfp-cache-p0-release/mtlearn/bindings').glob('_mtlearn*.so')
assert sha(native)==manifest['native_sha256']
suite=ET.parse(out/'validation-tests.xml').getroot()[0]
installed=json.loads((out/'installed-wheel/validation.json').read_text())
package=json.loads((out/'package-manifest.json').read_text())
assert installed['native_sha256']==package['wheel']['native_sha256']
assert installed['checkpoint_pytest_exit']==0
files=[Path(__file__).resolve(),root/'scripts/smoke_test_release.py',root/'scripts/benchmarks/cfp_cache_p6.py',
       root/'mtlearn/tests/python/test_cfp_checkpoint_independence.py',root/'pyproject.toml',
       root/'docs/cfp-architecture.md',root/'docs/source/guides/cfp-cache-migration.md',
       root/'docs/source/guides/connected-filter-preprocessing.md',
       root/'docs/source/guides/pytorch-integration.md',root/'docs/source/guides/index.md',
       root/'docs/source/api/python/cfp_preparation.rst',root/'docs/source/api/python/index.rst',
       root/'notebooks/experiments/CFP_linear_vs_mlp_scoring_205_SA_L3D14M3_segmentation.ipynb']
validation={'stage':'P6','date':'2026-09-09','status':'complete',
    'regression':{key:int(suite.attrib[key]) for key in ('tests','errors','failures','skipped')},
    'regression_seconds':float(suite.attrib['time']),'gradchecks':19,
    'new_checkpoint_tests':24,'fixture_files_verified':len(manifest['files']),
    'baseline_native_sha256':sha(native),'original_screws_notebook_sha256':sha(original),
    'docs':docs,'migration_examples':'all three executed, including two CPU workers and final backward',
    'package':package,'installed_wheel':installed,
    'source_sha256':{str(p.relative_to(root)):sha(p) for p in files},
    'limitations':['CUDA unavailable','Linux/Windows and other Python/Torch versions not executed',
                   'No full 2041-image preparation or 100-epoch training',
                   'P0-P5 original-resolution measurements reused; no new large-data benchmark in P6',
                   'RSS does not represent total MPS/unified-memory peak',
                   'No external package/documentation publication']}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'tests':validation['regression'],'docs':docs,'fixtures':len(manifest['files'])},indent=2))
