import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import numpy as np
import pytest
import torch
from llm.interpretation import interpret_instruction
from llm import openrouter
from training.training_utils import load_checkpoint
from ui.ui_study import StudyControls


def test_llm_interpretation_and_invalid_class_rejection(monkeypatch):
    result=dict(class_i=0,class_j=1,direction='away_from_j',scale='large',tighten_i=True,tighten_j=False)
    chat=MagicMock(return_value=(json.dumps(result),{}));monkeypatch.setattr(openrouter,'chat_completion',chat)
    fields,raw=interpret_instruction('Separate cats and dogs',{0:'cat',1:'dog'},result,model='test-model')
    assert fields==result
    assert 'Separate cats and dogs' in chat.call_args.args[0][1]['content']
    result['class_j']=99
    chat.return_value=(json.dumps(result),{})
    with pytest.raises(ValueError,match='unknown class'):interpret_instruction('move',{0:'cat',1:'dog'},dict(class_i=0,class_j=1))


def test_study_limits_block_apply_and_training():
    variable=lambda x:SimpleNamespace(get=lambda:x)
    panel=SimpleNamespace(study_mode_var=variable(True),study_started=None,study_count=0,
                          study_limit_var=variable(3),study_minutes_var=variable(5),status_var=MagicMock())
    assert not StudyControls.study_allows_apply(panel)
    import time
    panel.study_started=time.monotonic()
    assert StudyControls.study_allows_apply(panel)
    panel.study_count=3
    assert not StudyControls.study_allows_apply(panel)
    panel.study_count=0;panel.study_started=time.monotonic()-301
    assert not StudyControls.study_allows_apply(panel)


def test_checkpoint_loader_supports_weights_only_and_rolls_back_bad_weights(tmp_path):
    model=torch.nn.Linear(2,2);optimizer=torch.optim.Adam(model.parameters())
    original={k:v.clone() for k,v in model.state_dict().items()}
    good=tmp_path/'good.pt';torch.save(model.state_dict(),good)
    assert load_checkpoint(model,optimizer,good)==(0,{})
    bad=tmp_path/'bad.pt';torch.save({'weight':torch.zeros(3,2),'bias':torch.ones(2)},bad)
    with pytest.raises(RuntimeError):load_checkpoint(model,optimizer,bad)
    for k,v in model.state_dict().items():assert torch.equal(v,original[k])


def test_five_epoch_checkpoints_independent_of_pause_schedule(monkeypatch,tmp_path):
    from training import training as module
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,2)
            self.projection_layer=torch.nn.Linear(2,2)
            self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,x):
            f=self.layer(x);return f,f,f
    model=Model();optimizer=torch.optim.Adam(model.parameters())
    loader=[(torch.zeros(20,2),torch.zeros(20,dtype=torch.long))]
    save=MagicMock();pause=MagicMock();targets=MagicMock()
    monkeypatch.setattr(module,'save_checkpoint',save)
    monkeypatch.setattr(module,'save_report',MagicMock())
    monkeypatch.setattr(module,'evaluate_model',lambda *args:(50.,.5))
    monkeypatch.setattr(module,'compute_structure_for_strategy',lambda *args:None)
    get=lambda x:SimpleNamespace(get=lambda:x)
    module.train_model(model,optimizer,loader,loader,loader,torch.device('cpu'),6,100,
                       get(0.),str(tmp_path),pause_after_n_epochs=3,checkpoint_dir=str(tmp_path),
                       logger=MagicMock(),strategy_var=get(1),start_epoch=4,
                       epoch_end_callback=pause,target_callback=targets)
    assert [call.args[2] for call in save.call_args_list]==[5,10]
    assert pause.call_count==2  # epochs 6 and 9
    assert targets.call_count==3


def test_qsf_export_keeps_active_questions_and_both_scenario_branches(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'study'))
    from convert_qsf import convert
    schema=convert(output=tmp_path/'schema.json')
    assert len(schema['questions'])==26
    assert 'QID176' in schema['questions'] and 'QID192' in schema['questions']
    assert 'QID1' not in schema['questions']
    assert sum(n['Type']=='Branch' for n in schema['flow'])==2


def test_apply_logs_original_text_user_text_interpretation_and_movements(monkeypatch,tmp_path):
    import queue
    from ui import ui_llm
    from ui.ui_llm import LLMSuggestionsPanel
    from test_session_and_impact import session_ui,suggestion
    ui=session_ui(tmp_path)
    s=suggestion(suggestion='Move cat away from dog')
    ui.latest_llm_suggestions=[s];ui.applied_llm_suggestion_ids=set()
    ui.llm_tracker=MagicMock();ui.update_log=MagicMock()
    fields=dict(class_i=0,class_j=1,direction='away_from_j',scale='large',tighten_i=True,tighten_j=False)
    monkeypatch.setattr(ui_llm,'interpret_instruction',lambda *args,**kwargs:(fields,'test interpretation'))
    class Immediate:
        def __init__(self,target,**kwargs):self.target=target
        def start(self):self.target()
    monkeypatch.setattr(ui_llm.threading,'Thread',Immediate)
    panel=SimpleNamespace(ui=ui,text_interpretations={},text_flush={},model_var=SimpleNamespace(get=lambda:'test-model'),
                          _class_names=lambda:{0:'cat',1:'dog',2:'ship'},_suggestions=lambda:[s],
                          _log=ui.events.log,status_var=MagicMock(),result_queue=queue.Queue(),refresh_cards=MagicMock(),
                          text_setters={s.uid:MagicMock()})
    LLMSuggestionsPanel._set_text(panel,s,'Separate cat and dog a lot and tighten cat')
    _,payload=panel.result_queue.get_nowait()
    LLMSuggestionsPanel._interpretation_done(panel,*payload)
    panel.text_setters[s.uid].assert_called_with('Separate cat and dog a lot and tighten cat')
    assert LLMSuggestionsPanel._apply(panel,s)
    from training.training_utils import compute_structure_for_strategy
    from llm.strategies import get_strategy
    import matplotlib
    with monkeypatch.context() as backend_patch:
        backend_patch.setattr(matplotlib, 'use', lambda *args, **kwargs: None)
        from ui.ui import UI
    ui.plot.get_moved_2d_points=lambda:ui.moved_points
    ui.plot.samples_per_class=int(np.sum(ui.data['labels']==0))
    ui.plot.num_classes=len(np.unique(ui.data['labels']))
    ui.alpha_var=SimpleNamespace(get=lambda:.5)
    target=compute_structure_for_strategy(get_strategy(4),ui.plot,torch.device('cpu'),[s])
    UI.record_training_target(ui,target,get_strategy(4),1)
    ui.events.close()
    events=[json.loads(line) for line in Path(ui.events.path).read_text().splitlines()]
    types=[e['type'] for e in events]
    assert 'SUGGESTION_TEXT_CHANGED' in types and 'TEXT_INTERPRETATION' in types
    applied=next(e for e in events if e['type']=='SUGGESTION_APPLY_REQUESTED')
    assert applied['original_text']=='Move cat away from dog'
    assert applied['user_text']=='Separate cat and dog a lot and tighten cat'
    movement=next(e for e in events if e['type']=='MODIFICATION' and e['kind']=='llm_apply')
    assert movement['vectors'] and movement['state_after']['points']
    assert movement['suggestion']['interpretation']['fields']['scale']=='large'
    target_event=next(e for e in events if e['type']=='TRAINING_TARGET_BUILT')
    assert target_event['sources'][0]['original_text']=='Move cat away from dog'
    assert target_event['sources'][0]['suggestion']=='Separate cat and dog a lot and tighten cat'
    for c in np.unique(ui.data['labels']):
        np.testing.assert_allclose(target_event['target'][str(c)]['center'],
                                   ui.moved_points[ui.data['labels']==c].mean(axis=0),atol=1e-6)


def test_study_guard_covers_auto_apply_and_counts_committed_movement_once():
    from test_overlay_drag import plot_ui,suggestion
    from ui.ui_display import apply_llm_suggestion
    s=suggestion();ui,points,labels=plot_ui([s])
    ui.study_allows_apply=lambda:False
    ui.record_study_interaction=MagicMock()
    assert not apply_llm_suggestion(ui,s,source='auto')
    np.testing.assert_allclose(ui.moved_points,points)
    ui.record_study_interaction.assert_not_called()
    ui.study_allows_apply=lambda:True
    assert apply_llm_suggestion(ui,s,source='auto')
    ui.record_study_interaction.assert_called_once_with(s)
    import matplotlib.pyplot as plt
    plt.close(ui.scatter_fig)


def test_study_rejects_pause_request_until_training_acknowledges_it():
    import threading
    from ui.ui_study import training_is_idle_or_paused
    pause,ack=threading.Event(),threading.Event()
    pause.set()
    ui=SimpleNamespace(training_thread=SimpleNamespace(is_alive=lambda:True),pause_event=pause,
                       training_paused=ack,study_mode_var=MagicMock(),status_var=MagicMock())
    ui.study_mode_var.get.return_value=True
    assert not training_is_idle_or_paused(ui)
    StudyControls.toggle_study_mode(ui)
    ui.study_mode_var.set.assert_called_once_with(False)
    ack.set()
    assert training_is_idle_or_paused(ui)
    pause.clear()
    assert not training_is_idle_or_paused(ui)


def test_confirmed_pause_allows_study_without_resuming_training(monkeypatch,tmp_path):
    import threading
    from training import training as module
    from ui.ui_study import training_is_idle_or_paused
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,2)
            self.projection_layer=torch.nn.Linear(2,2)
            self.scale=torch.nn.Parameter(torch.tensor(1.));self.calls=0
        def forward(self,x):
            self.calls+=1;f=self.layer(x);return f,f,f
    model=Model();optimizer=torch.optim.Adam(model.parameters())
    pause,ack,stop=threading.Event(),threading.Event(),threading.Event();pause.set()
    marker={'value':'before'};seen=[];errors=[]
    monkeypatch.setattr(module,'compute_structure_for_strategy',lambda *args:seen.append(marker['value']))
    monkeypatch.setattr(module,'save_report',MagicMock())
    monkeypatch.setattr(module,'evaluate_model',lambda *args:(50.,.5))
    get=lambda x:SimpleNamespace(get=lambda:x)
    loader=[(torch.zeros(20,2),torch.zeros(20,dtype=torch.long))]
    def train():
        try:
            module.train_model(model,optimizer,loader,loader,loader,torch.device('cpu'),1,100,get(0.),
                               str(tmp_path),pause_after_n_epochs=10,logger=MagicMock(),
                               strategy_var=get(1),pause_event=pause,paused_state=ack,stop_training=stop)
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=train,daemon=True);thread.start()
    try:
        assert ack.wait(3)
        assert model.calls==0 and seen==[]
        state={'enabled':True}
        ui=SimpleNamespace(training_thread=thread,pause_event=pause,training_paused=ack,
            study_mode_var=SimpleNamespace(get=lambda:state['enabled']),training_button=MagicMock(),
            study_config_widgets=[],events=MagicMock(),update_study_status=MagicMock())
        assert training_is_idle_or_paused(ui)
        StudyControls.toggle_study_mode(ui)
        ui.training_button.configure.assert_called_with(state='disabled')
        assert pause.is_set() and ack.is_set() and model.calls==0
        marker['value']='study edits'
        state['enabled']=False;StudyControls.toggle_study_mode(ui)
        assert pause.is_set()  # disabling study alone must not resume training
        pause.clear();thread.join(5)
        assert not thread.is_alive() and not errors
        assert seen==['study edits'] and model.calls==1
    finally:
        stop.set();pause.clear();thread.join(5)


def test_study_task_can_use_current_trained_epoch_without_loading_file(tmp_path):
    import threading
    state=SimpleNamespace(study_mode_var=SimpleNamespace(get=lambda:True),loaded_checkpoint=None,
                          latest_metrics={'epoch':5},training_thread=SimpleNamespace(is_alive=lambda:True),
                          pause_event=threading.Event(),training_paused=threading.Event(),status_var=MagicMock(),
                          study_limit_var=SimpleNamespace(get=lambda:3),study_minutes_var=SimpleNamespace(get=lambda:5),
                          study_config_widgets=[],probant_scenario_dir=str(tmp_path),events=MagicMock(),
                          update_study_status=MagicMock())
    state.pause_event.set();state.training_paused.set()
    StudyControls.begin_study_task(state)
    assert state.study_count==0 and state.study_started is not None
    data=state.events.log.call_args.kwargs
    assert data['epoch']==5 and data['model_source']=='paused_training'


def test_scheduled_epoch_pause_acknowledges_and_rebuilds_after_study(monkeypatch,tmp_path):
    import threading
    from training import training as module
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,2);self.projection_layer=torch.nn.Linear(2,2)
            self.scale=torch.nn.Parameter(torch.tensor(1.));self.calls=0
        def forward(self,x):
            self.calls+=1;f=self.layer(x);return f,f,f
    model=Model();optimizer=torch.optim.Adam(model.parameters())
    pause,ack,stop=threading.Event(),threading.Event(),threading.Event()
    marker={'value':'before'};seen=[];errors=[];epochs=[]
    monkeypatch.setattr(module,'compute_structure_for_strategy',lambda *args:seen.append(marker['value']))
    monkeypatch.setattr(module,'save_report',MagicMock())
    monkeypatch.setattr(module,'evaluate_model',lambda *args:(50.,.5))
    get=lambda x:SimpleNamespace(get=lambda:x)
    loader=[(torch.zeros(20,2),torch.zeros(20,dtype=torch.long))]
    def after_epoch():
        if len(epochs)==2:stop.set()
    def train():
        try:
            module.train_model(model,optimizer,loader,loader,loader,torch.device('cpu'),2,100,get(0.),str(tmp_path),
                               pause_after_n_epochs=1,logger=MagicMock(),strategy_var=get(1),pause_event=pause,
                               paused_state=ack,stop_training=stop,metrics_callback=epochs.append,
                               epoch_end_callback=after_epoch)
        except Exception as exc:errors.append(exc)
    worker=threading.Thread(target=train,daemon=True);worker.start()
    try:
        assert ack.wait(3)
        assert pause.is_set() and len(epochs)==1 and model.calls==1
        marker['value']='study edits';pause.clear();worker.join(5)
        assert not worker.is_alive() and not errors
        assert model.calls==2 and seen==['before','study edits']
    finally:
        stop.set();pause.clear();worker.join(5)
