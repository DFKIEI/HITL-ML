import torch
import pytest
from training.losses import relative_distance_loss


def reference(features,labels,target):
    total=features.sum()*0;terms=0
    groups={c:features[labels==c] for c in target}
    for c,points in groups.items():
        if not len(points):continue
        center=points.mean(0);spread=(points-center).norm(dim=1).mean()
        total=total+.5*((center-target[c]['center'])**2).sum()*(2/features.shape[1])**.5+(spread-target[c]['spread']).abs();terms+=2
        for other,others in groups.items():
            if c!=other and len(others):
                total=total+(target[c]['spread']+target[other]['spread']-(center-others.mean(0)).norm()).clamp_min(0);terms+=1
    return total/max(1,terms)


@pytest.mark.parametrize('dimension',[2,64])
@pytest.mark.parametrize('labels',[[0,0,1,1,2,2],[0,0,2,2],[1]])
def test_vectorized_loss_and_gradients_match_original(dimension,labels):
    torch.manual_seed(4)
    x=torch.randn(len(labels),dimension,dtype=torch.float64,requires_grad=True)
    y=torch.tensor(labels)
    target={c:dict(center=torch.randn(dimension,dtype=torch.float64),spread=torch.tensor(.7,dtype=torch.float64)) for c in range(3)}
    expected=reference(x,y,target);actual=relative_distance_loss(x,y,target)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])


def test_evaluation_keeps_all_output_classes_even_when_batch_is_missing_some():
    from training.training import evaluate_model
    class Classifier(torch.nn.Module):
        def forward(self,inputs):
            logits=torch.tensor([[0.,1.,5.]]).repeat(len(inputs),1)
            return logits,inputs,inputs
    accuracy,_=evaluate_model(Classifier(),[(torch.ones(2,2),torch.zeros(2,dtype=torch.long))],torch.device('cpu'))
    assert accuracy==0.0


def test_headless_pretraining_saves_epoch_25_without_plot_or_discarding_small_batch(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from training.training import train_model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer=torch.nn.Linear(2,2)
            self.projection_layer=torch.nn.Linear(2,2)
            self.scale=torch.nn.Parameter(torch.tensor(1.))
            self.batch_sizes=[]
        def forward(self,x):
            self.batch_sizes.append(len(x));features=self.layer(x)
            return features,features,features
    model=Model();loader=[(torch.randn(4,2),torch.tensor([0,1,0,1]))]
    metrics=[]
    train_model(model,torch.optim.Adam(model.parameters()),loader,loader,loader,torch.device('cpu'),
                1,100,SimpleNamespace(get=lambda:0.),str(tmp_path),checkpoint_dir=str(tmp_path),
                logger=MagicMock(),metrics_callback=metrics.append,start_epoch=24)
    checkpoint=torch.load(tmp_path/'checkpoint_epoch_25.pt',weights_only=False)
    assert checkpoint['epoch']==25
    assert model.batch_sizes==[4,4,4]
    assert metrics[0]['train_samples_per_second']>0


def test_geometry_remains_float32_under_autocast():
    x=torch.randn(4,2)
    target={c:dict(center=torch.zeros(2),spread=torch.tensor(1.)) for c in range(2)}
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        result=relative_distance_loss(x,torch.tensor([0,0,1,1]),target)
    assert result.dtype==torch.float32
