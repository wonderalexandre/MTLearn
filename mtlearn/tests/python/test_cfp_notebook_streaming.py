"""P5 notebook metrics against the independent pre-migration concatenation path."""
import ast
import hashlib
import json
import math
from pathlib import Path
import weakref

import numpy as np
import pytest

nbformat = pytest.importorskip("nbformat")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer
from mtlearn.layers.cfp import CFPPreprocessor, DiskStore, PreparedDataset, collate_prepared

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / 'notebooks/experiments/CFP_linear_vs_mlp_scoring_205_SA_L3D14M3_segmentation.ipynb'
REFERENCE = Path(__file__).parent / 'fixtures/cfp_cache_p5/notebook_before_p5.json'
if not NOTEBOOK.exists() or not REFERENCE.exists():
    pytest.skip('Notebook regression requires the repository checkout and P5 reference', allow_module_level=True)
BEFORE = json.loads(REFERENCE.read_text())
GRID = [round(v, 2) for v in np.linspace(.05, .95, 91)]


def definitions(source, namespace):
    nodes = [node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


def environment():
    return dict(torch=torch, np=np, F=F, math=math, POSITIVE_BCE_WEIGHT=7.5,
        DICE_LOSS_WEIGHT=.5, BCE_LOSS_WEIGHT=.5, THRESHOLD_GRID=GRID, RESPONSE_THRESHOLD=.5,
        OUTPUT_SCALE=255., EVAL_THRESHOLD_BLOCK_SIZE=8, EVAL_PIXEL_BLOCK_SIZE=37,
        device='cpu', LEARNING_RATE=.01)


def current():
    ns=environment()
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code' and 'class SegmentationAccumulator' in cell.source:
            definitions(cell.source,ns)
    return ns


def legacy():
    return definitions(BEFORE['cells']['13'],environment())


@pytest.mark.parametrize('batch_size',[1,2,3,5])
@pytest.mark.parametrize('block',[1,7,100])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_grid_metrics_losses_and_last_batch_match_concatenation(batch_size,block,dtype):
    new,old=current(),legacy()
    rng=torch.Generator().manual_seed(15)
    responses=torch.rand((5,1,6,7),generator=rng,dtype=dtype)
    targets=(torch.rand((5,1,6,7),generator=rng)>.8).to(dtype)
    # Include exact boundaries and constants, including empty/full masks.
    responses[0]=.5
    responses[1,0,0,:4]=torch.tensor([.05,.95,0,1],dtype=dtype)
    targets[0]=0;targets[1]=1
    accumulator=new['SegmentationAccumulator'](GRID,threshold_block_size=block,pixel_block_size=11)
    for begin in range(0,5,batch_size): accumulator.update(responses[begin:begin+batch_size],targets[begin:begin+batch_size])
    assert accumulator.image_count==5 and accumulator.pixel_count==responses.numel()
    for index,threshold in enumerate(GRID):
        assert accumulator.metrics(index)==old['metrics_from_responses'](responses,targets,threshold)
    assert accumulator.best_f1()==old['best_f1_threshold'](responses,targets)
    assert accumulator.loss==pytest.approx(old['segmentation_loss'](responses,targets).item(),rel=1e-6,abs=1e-7)


def test_unsorted_duplicate_thresholds_ties_and_dtype_boundaries():
    new,old=current(),legacy()
    grid=[.9,.2,.2,.5,.05]
    r=torch.tensor([[[[0.,.05,.2,.5,.9,1.]]]])
    t=torch.zeros_like(r)
    accumulator=new['SegmentationAccumulator'](grid,threshold_block_size=2,pixel_block_size=2)
    accumulator.update(r,t)
    assert accumulator.best_f1()['threshold']==.9  # First tied candidate, not sorted threshold.
    assert accumulator.best_f1(candidate_count=1)==accumulator.metrics(0)
    for i,threshold in enumerate(grid): assert accumulator.metrics(i)==old['metrics_from_responses'](r,t,threshold)
    assert torch.equal(accumulator.counts[1],accumulator.counts[2])


def test_different_shapes_weight_dice_by_images_and_bce_by_pixels():
    ns=current()
    first=torch.tensor([[[[.1,.8]]]])
    second=torch.full((1,1,5,7),.4)
    targets=[torch.tensor([[[[0.,1.]]]]),torch.zeros_like(second)]
    accumulator=ns['SegmentationAccumulator']([.5],pixel_block_size=4)
    for r,t in zip([first,second],targets): accumulator.update(r,t)
    dice=sum(ns['soft_dice_loss'](r,t).item() for r,t in zip([first,second],targets))/2
    bce=sum(ns['balanced_bce_loss'](r,t).item()*r.numel() for r,t in zip([first,second],targets))/(first.numel()+second.numel())
    assert accumulator.loss==.5*dice+.5*bce


def test_accumulator_retention_independent_of_number_of_samples(monkeypatch):
    ns=current()
    accumulator=ns['SegmentationAccumulator'](GRID,threshold_block_size=5,pixel_block_size=11)
    def no_cat(*args,**kwargs): raise AssertionError('Do not concatenate a dataset of responses')
    monkeypatch.setattr(torch,'cat',no_cat)
    for _ in range(101):
        response=torch.rand((1,1,3,4));target=torch.zeros_like(response)
        refs=(weakref.ref(response),weakref.ref(target))
        accumulator.update(response,target)
        del response,target
        assert all(ref() is None for ref in refs)
    assert accumulator.counts.shape==(len(GRID),4)
    assert sum(value.numel()*value.element_size() for value in vars(accumulator).values() if torch.is_tensor(value))==len(GRID)*4*8
    assert accumulator.image_count==101


def test_eval_restores_mode_and_releases_each_response_on_error_and_success():
    ns=current()
    class Model(torch.nn.Module):
        def __init__(self): super().__init__();self.previous=None
        def forward(self,x):
            assert self.previous is None or self.previous() is None
            result=x.clone();self.previous=weakref.ref(result)
            return result
    model=Model()
    loader=[(torch.arange(6).reshape(1,1,2,3).float(),torch.zeros(1,1,2,3)) for _ in range(3)]
    result=ns['evaluate_streaming'](model,loader,[.5])
    assert model.training and result.image_count==3 and model.previous() is None
    model.eval()
    with pytest.raises(ValueError): ns['evaluate_streaming'](model,[],[.5])
    assert not model.training
    model.train()
    broken=[(torch.full((1,1,2,3),float('nan')),torch.zeros(1,1,2,3))]
    with pytest.raises(ValueError): ns['evaluate_streaming'](model,broken,[.5])
    assert model.training


def test_threshold_selection_is_train_only_and_reporting_schema_preserved():
    ns=current()
    calls=[]
    class Summary:
        image_count=1
        loss=.2
        def best_f1(self,candidate_count):
            assert candidate_count==len(GRID)
            return self.metrics(GRID.index(.35))
        def metrics(self,index):
            threshold=self.thresholds[index]
            return {'threshold':threshold,'f1':.3,'iou':.2,'precision':.4,'recall':.25,'accuracy':.8,'foreground_fraction':.1}
    def evaluate(layer,loader,thresholds):
        calls.append((loader,list(thresholds)))
        result=Summary();result.thresholds=thresholds
        return result
    ns.update(evaluate_streaming=evaluate,prepared_loaders={'train':'train-only','test':'test-only'},
              parameter_summary=lambda layer:{'num_parameters':2,'parameter_l2':1},selected_thresholds={})
    result=ns['collect_metrics'](1,'model',None,train_loss=.7)
    assert calls==[('train-only',GRID+[.5]),('test-only',[.35,.5])]
    assert result['train_threshold']==result['test_threshold']==result['threshold']==.35
    assert result['train_epoch_loss']==.7 and ns['selected_thresholds']=={'model':.35}


def test_original_protocol_dataset_and_loss_code_preserved():
    notebook=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(notebook)
    before=ast.parse(BEFORE['cells']['7'])
    after=ast.parse(notebook.cells[7].source)
    def assigned(tree):
        return {n.targets[0].id:ast.dump(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    original=assigned(before);actual=assigned(after)
    for name,value in original.items():
        # The user reduced MAX_SAMPLES to fit the SSD after the P5 migration.
        if name not in ('config', 'MAX_SAMPLES'): assert actual[name]==value,name
    assert ast.dump(ast.parse(BEFORE['cells']['9']).body[0])==ast.dump(ast.parse(notebook.cells[9].source).body[0])
    # Dataset indexing and seeded split before preview are unchanged.
    assert BEFORE['cells']['9'].split('image_in, image_target, name =')[0]==notebook.cells[9].source.split('image_in, image_target, name =')[0]
    for name in ('normalize_per_image','soft_dice_loss','balanced_bce_loss','segmentation_loss'):
        def node(source): return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(node(BEFORE['cells']['13']))==ast.dump(node(notebook.cells[13].source)),name
    original_screws=NOTEBOOK.with_name('CFP_linear_vs_mlp_scoring_screws_segmentation.ipynb')
    source_before_rename = original_screws.read_bytes().replace(
        b'Type.GRAY_LEVEL_HEIGHT', b'Type.GRAY_HEIGHT')
    assert hashlib.sha256(source_before_rename).hexdigest()==BEFORE['source_screws_sha256']
    assert 'collect_responses_and_targets' not in '\n'.join(c.source for c in notebook.cells)


@pytest.mark.parametrize('bad', [dict(thresholds=[]),dict(thresholds=[float('nan')]),
    dict(thresholds=[.5],pixel_block_size=0),dict(thresholds=[.5],threshold_block_size=True)])
def test_invalid_accumulator_options(bad):
    with pytest.raises(ValueError): current()['SegmentationAccumulator'](**bad)


@pytest.mark.integration
@pytest.mark.parametrize('batch_size',[1,2])
def test_shared_ssd_preserves_legacy_initialization_and_one_epoch(tmp_path,batch_size):
    ns=current();old=legacy()
    images=torch.rand((5,1,6,7),generator=torch.Generator().manual_seed(12))
    targets=(images>.7).float()
    train=TensorDataset(images[:3],targets[:3]);test=TensorDataset(images[3:],targets[3:])
    def models():
        return {kind:Layer(1,[{'name':kind,'tree_type':morphology.TreeType.MAX_TREE,
            'attributes':[morphology.AttributeType.AREA,morphology.AttributeType.GRAY_LEVEL_HEIGHT],
            'scoring': {'kind':'linear_sigmoid'} if kind=='linear' else {'kind':'mlp','hidden_units':[8],'activation':'tanh'}}],
            scale_mode='dataset_clipped_zscore01',clamp=12.) for kind in ('linear','mlp')}
    torch.manual_seed(42);previous=models()
    legacy_loaders={}
    for kind,layer in previous.items():
        legacy_loaders[kind]=layer.build_dataloader_cached(DataLoader(train,batch_size=batch_size,shuffle=True))
        layer.build_dataloader_cached_fixed_stats(DataLoader(test,batch_size=batch_size),index_offset=len(train))
        layer.init_identity(p0=.995)
    torch.manual_seed(42);updated=models()
    prep=CFPPreprocessor.from_layer(updated['linear'])
    with DiskStore(tmp_path,max_disk_bytes=1024**2) as store:
        result=prep.prepare(train,store=store,manifest='train',source_version='v1',preprocessing_version='v1',collect_stats=True)
        loader=DataLoader(PreparedDataset(store,'train',train),batch_size=batch_size,collate_fn=collate_prepared,shuffle=False)
        for kind,layer in updated.items():
            layer.set_stats(result.statistics)
            for _ in range(2): torch.empty((),dtype=torch.int64).random_()
            layer.init_identity(p0=.995)
            for (_,a),(_,b) in zip(layer.named_parameters(),previous[kind].named_parameters()): assert torch.equal(a,b)
        # Compare the actual notebook training cell, with reports disabled, to
        # the unchanged pre-P5 training loop on the same small fixed source.
        def train_cell(layers,namespace,source,is_new):
            namespace.update(cfp_layers=layers,torch=torch,time=__import__('time'),pd=__import__('pandas'),
                LEARNING_RATE=.01,WEIGHT_DECAY=1e-7,GRAD_CLIP=5.,NUM_EPOCHS=1,REPORT_EPOCHS=set(),initial_rows=[])
            if is_new: namespace['prepared_loaders']={'train':loader}
            else: namespace['cached_loaders']={kind:{'train':legacy_loaders[kind]} for kind in layers}
            exec(compile(source,str(NOTEBOOK),'exec'),namespace)
        notebook=nbformat.read(NOTEBOOK,as_version=4)
        train_cell(previous,old,BEFORE['cells']['17'],False)
        train_cell(updated,ns,notebook.cells[17].source,True)
        for kind,layer in updated.items():
            for name,parameter in layer.named_parameters():
                torch.testing.assert_close(parameter,dict(previous[kind].named_parameters())[name],rtol=1e-5,atol=1e-6)
            assert layer.cached_sample_count()==0
        assert store.info()['writes']==3
